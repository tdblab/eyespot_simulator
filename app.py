import streamlit as st
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import datetime
import base64
import zipfile
import io
from PIL import Image

# =====================================================================
# MODEL
# =====================================================================
#   Fz4 (competency field), repressed by Wg:
#       d[Fz4]/dt = beta_f (y/Ly) / (1 + ([Wg]/Kwg)^n_wg) - k_f [Fz4]
#
#   Local competency of the Dll-dependent reaction:
#       r_eff(x,y) = r_max / (1 + ([Fz4]/Kfz)^n_gate)
#
#   Kfz must sit BELOW max[Fz4] = beta_f/k_f for the gate to be a
#   competency boundary rather than a gentle ramp; n_gate sets how
#   sharp that boundary is. The boundary lies at y = Kfz/max[Fz4] * Ly.
#
#   Dll is algebraic, with Spalt feedback on focal Dll
#   (Matsuoka & Monteiro 2022: sal regulates focal Dll in hindwings):
#       [Dll] = (k1l/k3) [Wg] (1 + f_s S^n_sd / (K_sd^n_sd + S^n_sd))
#
#   Core Gray-Scott pair, r_eff gating BOTH sides (mass conserved):
#       d[Wg]/dt  = r_eff [Dll]^2 [Dpp] - k1 [Wg] + D1 lap[Wg]
#
#   Diffusion is isotropic (rho_wg = 1), as in Connahs et al. A circular
#   focus follows from their initial condition (no substrate at t = 0),
#   not from any anisotropy.
#       d[Dpp]/dt = alpha(t) - r_eff [Dll]^2 [Dpp] - k2 [Dpp] + D2 lap[Dpp]
#
#   Downstream readouts (no feedback onto the core):
#       U_S = [Dll]^3        (Dll is slaved to Wg, so [Dll]^2[Wg] was the
#                             same quantity written twice; the cube supplies
#                             cooperativity, not a second input)
#       d[S]/dt    = beta_s U_S^n/(Ks^n+U_S^n)(1 + Antp^n/(Kas^n+Antp^n)) - k_s[S]
#       U_A = [Dll][S],  X(y) = X0 exp(-y/lambda_X)
#       d[Antp]/dt = beta_a U_A^n/(Ka^n+U_A^n) / (1+(X/K_x)^n_rep) - k_a[Antp]
#
#   y/Ly is ZERO at the wing margin and maximal proximally, so Fz4 is low
#   at the margin and high proximally: r_eff is maximal at the margin and
#   clamped down proximally. That proximal ceiling is what stops the
#   activator finger extending and splitting in wild type.
# =====================================================================

st.set_page_config(page_title="Eyespot Simulator", page_icon="🦋",
                   layout="wide")

st.title("🦋 Bicyclus anynana Eyespot Simulator")
st.markdown("""
Interactive *in silico* CRISPR clone engine.
Gray--Scott activator--substrate kinetics with Dll inside the non-linear
reaction term; a single global `r_max` is locally gated by Fz4.
""")

# =====================================================================
# SIDEBAR
# =====================================================================
with st.sidebar:
    run_button = st.button("🚀 Run Simulation", type="primary",
                           use_container_width=True)

    st.markdown("---")
    st.header("🧬 CRISPR Setup")

    clone_shape = st.selectbox(
        "Somatic Clone Shape",
        options=['None', 'Center', 'Diagonal', 'Corner', 'Full'],
        index=1
    ).lower()

    target_genes = st.multiselect(
        "Select Genes to Knockout in Clone",
        options=['Wg', 'Dpp', 'Dll', 'Fz4', 'S', 'Antp'],
        default=['Dll']
    )

    # only the center and corner clones have a width; diagonal and full
    # clones are defined by geometry alone
    uses_width = clone_shape in ('center', 'corner')
    clone_width = st.slider("Clone Width (Pixels)", min_value=2, max_value=40,
                            value=2, step=2, disabled=not uses_width,
                            help="2 px = 5 um. The centre-clone split is "
                                 "width-sensitive; see the note in the app.")
    Fz4_knockout = st.checkbox("Global Fz4 Knockout (Whole Tissue)",
                               value=False,
                               help="Mechanistic: Fz4 = 0 lifts the gate, so "
                                    "r_eff rises to r_max everywhere.")

    st.markdown("---")
    st.header("⚙️ Network Parameters")

    r_max = st.number_input("Max Coupling (r_max)", value=1.45e-7,
                            format="%.2e",
                            help="Uninhibited coupling. One global value for "
                                 "every genotype.")

    st.markdown("---")

    with st.expander("1. Core Morphogens (Wg & Dpp)", expanded=False):
        D1 = st.number_input("Wg Diffusion (D1)", value=0.01, format="%.4f")
        D2 = st.number_input("Dpp Diffusion (D2)", value=0.12, format="%.4f")
        rho_wg = st.slider(
            "Wg diffusion anisotropy (rho = D_y/D_x)", 0.20, 1.00, 1.00, 0.05,
            help="1.00 = isotropic, as in Connahs et al. Not needed for a "
                 "circular focus once the substrate starts from zero; kept "
                 "only as an exploratory knob.")
        k1 = st.number_input("Wg Degradation (k1)", value=0.1e-3, format="%.2e")
        k2 = st.number_input("Dpp Degradation (k2)", value=0.08e-3, format="%.2e")
        alpha = st.number_input("Dpp Production (alpha)", value=6.2e-3,
                                format="%.2e")
        alpha_late_mult = st.slider("Maturation Drop (alpha_late fraction)",
                                    0.50, 1.00, 0.75, 0.01)
        T_late_days = st.number_input("Maturation Time (Days)", value=2.50,
                                      step=0.1)
        alpha_ramp_h = st.slider("Maturation Ramp Width (hours)", 0.0, 24.0,
                                 0.0, 0.5,
                                 help="0 = abrupt step. A few hours removes "
                                      "the transient the step produces.")

    with st.expander("2. Receptors (Fz4) & Transducers (Dll)", expanded=False):
        k1l = st.number_input("Dll Activation (k1l)", value=9.98e-4, format="%.2e")
        k3 = st.number_input("Dll Degradation (k3)", value=1.0e-3, format="%.2e")
        f_s = st.number_input(
            "Spalt feedback on Dll (f_s)", value=0.40, step=0.05, format="%.2f",
            help="Maximal fractional boost of focal Dll by Spalt "
                 "(Matsuoka & Monteiro 2022). 0 = feed-forward model. "
                 "f_s >= 0.35 is needed for the fz4 knockout to duplicate.")
        K_sd = st.number_input(
            "Spalt level for half-maximal Dll boost (K_sd)", value=300.0,
            step=25.0)
        n_sd = st.slider("Hill coefficient, Spalt -> Dll (n_sd)",
                         1.0, 4.0, 2.0, 0.5)
        beta_f = st.number_input("Fz4 Production (beta_f)", value=1.2, step=0.1)
        k_f = st.number_input("Fz4 Degradation (k_f)", value=1.0e-3, format="%.2e")
        Kwg = st.number_input(
            "Wg Repression of Fz4 (Kwg)", value=80.0, step=1.0,
            help="Sets the depth of the Fz4 HCR hole. LOWER = deeper hole but "
                 "stronger Wg->Fz4->r_eff positive feedback. At 40 the wild "
                 "type splits into two stacked spots; 50 gives a ~76% deep "
                 "hole with a single WT spot.")
        Kfz = st.number_input(
            "Fz4 Inhibition of r_eff (Kfz)", value=450.0, step=25.0,
            help="Competency boundary. Must be BELOW max[Fz4] = beta_f/k_f "
                 "(= 1200 at defaults), otherwise the gate never engages and "
                 "acts as a shallow ramp. Boundary sits at "
                 "y = Kfz/max[Fz4] x Ly, i.e. 98 um at Kfz = 450. With n_gate = 2 "
                 "this is a graded boundary rather than a step.")
        n_gate = st.slider(
            "Gate sharpness, Fz4 -> r_eff (n_gate)", 1.0, 10.0, 2.0, 0.5,
            help="High values turn the gate into a step, which confines the "
                 "focus distally and rounds it. 2.0 reproduces the old "
                 "gradual ramp.")
        n_wg = st.slider(
            "Wg repression of Fz4 (n_wg)", 1.0, 5.0, 2.0, 0.1,
            help="Separate from n_gate: sets the sharpness of the Fz4 hole "
                 "at the eyespot centre.")

    with st.expander("3. Downstream TFs (Spalt & Antp)", expanded=False):
        beta_s = st.number_input("Spalt Production (beta_s)", value=2.8, step=0.1)
        beta_a = st.number_input("Antp Production (beta_a)", value=12.0, step=0.5)
        k_s = st.number_input("Spalt Degradation (k_s)", value=1.70e-3, format="%.2e")
        k_a = st.number_input("Antp Degradation (k_a)", value=1.80e-3, format="%.2e")
        Ks = st.number_input("Spalt Activation Threshold (Ks)", value=8.0e5,
                             format="%.2e")
        Ka = st.number_input("Antp Activation Threshold (Ka)", value=6.0e4,
                             format="%.2e")
        Kas = st.number_input("Antp Feedback Threshold (Kas)", value=200.0,
                              step=10.0)
        n_hill = st.slider("TF Hill Coefficient (n_hill)", 1.0, 5.0, 3.0, 0.1)

    with st.expander("4. Margin Repressor X(y)", expanded=False):
        X0 = st.number_input("X amplitude (X0)", value=10.0, step=1.0)
        lambda_X = st.number_input("X decay length (um)", value=40.0, step=5.0)
        K_x = st.number_input("X Repression Threshold (K_x)", value=2.0, step=0.5)
        n_rep = st.slider("X Hill Coefficient (n_rep)", 1.0, 5.0, 2.0, 0.1)

# =====================================================================
# GRID
# =====================================================================
dx = 2.5
dt = 1.8
nx, ny = 60, 105
Lx, Ly = nx * dx, ny * dx          # 150.0 x 262.5 um, consistent with grid
cmargin = 65.0


@jax.jit
def laplacian(Z, rho=1.0):
    """5-point Laplacian with replicated ghost cells; rho = D_y / D_x."""
    P = jnp.pad(Z, 1, mode='edge')
    d2x = (P[1:-1, :-2] + P[1:-1, 2:] - 2.0 * Z) / (dx ** 2)
    d2y = (P[:-2, 1:-1] + P[2:, 1:-1] - 2.0 * Z) / (dx ** 2)
    return d2x + rho * d2y


def build_clone_mask(shape, width):
    mask = np.zeros((ny, nx), dtype=bool)
    if shape == 'center':
        cx = nx // 2
        mask[:, cx - (width // 2): cx + (width // 2)] = True
    elif shape == 'diagonal':
        for i in range(ny):
            for j in range(nx):
                if i > ny - (j * (ny / nx)) - (ny // 4):
                    mask[i, j] = True
    elif shape == 'corner':
        mask[ny - (width * 2):, :width * 2] = True
    elif shape == 'full':
        mask[:, :] = True
    return jnp.array(mask)


# =====================================================================
# MAIN SIMULATION
# =====================================================================
if run_button:
    alpha_late = alpha_late_mult * alpha
    T_late_sec = T_late_days * 24 * 3600
    ramp_sec = max(alpha_ramp_h * 3600.0, 1e-9)

    total_days = 6.0
    total_steps = int((total_days * 24 * 3600) / dt)
    num_frames = 60
    steps_per_frame = total_steps // num_frames

    y_vals = jnp.linspace(0, Ly, ny)
    x_vals = jnp.linspace(0, Lx, nx)
    Y, X_grid = jnp.meshgrid(y_vals, x_vals, indexing='ij')

    Y_gradient = Y / Ly                    # 0 at the margin, 1 proximally
    X_repressor = X0 * jnp.exp(-Y / lambda_X)

    vein_mask = jnp.zeros((ny, nx), dtype=bool)
    vein_mask = vein_mask.at[:, 0].set(True)
    vein_mask = vein_mask.at[-1, :].set(True)
    vein_mask = vein_mask.at[:, -1].set(True)

    clone_mask_jnp = build_clone_mask(clone_shape, clone_width)

    mask_Wg = clone_mask_jnp if 'Wg' in target_genes else jnp.zeros_like(clone_mask_jnp)
    mask_Dpp = clone_mask_jnp if 'Dpp' in target_genes else jnp.zeros_like(clone_mask_jnp)
    mask_Dll = clone_mask_jnp if 'Dll' in target_genes else jnp.zeros_like(clone_mask_jnp)
    mask_Fz4 = clone_mask_jnp if 'Fz4' in target_genes else jnp.zeros_like(clone_mask_jnp)
    mask_S = clone_mask_jnp if 'S' in target_genes else jnp.zeros_like(clone_mask_jnp)
    mask_Antp = clone_mask_jnp if 'Antp' in target_genes else jnp.zeros_like(clone_mask_jnp)

    @jax.jit
    def update_n_steps(state, start_time_sec):
        times = start_time_sec + jnp.arange(steps_per_frame) * dt

        def step_fn(carry, t):
            Wg, Dpp, S, Antp, Fz4 = carry

            # 1. Dll: algebraic readout of Wg, boosted by Spalt at the focus;
            #    zero inside a Dll-null clone
            sal_boost = 1.0 + f_s * (S ** n_sd) / (K_sd ** n_sd + S ** n_sd + 1e-8)
            Dll = (k1l / k3) * Wg * sal_boost
            Dll = jnp.where(mask_Dll, 0.0, Dll)

            # 2. Local competency field
            r_eff = r_max / (1.0 + (Fz4 / Kfz) ** n_gate)

            # 3. Core reaction, gating production and consumption alike
            turing_reaction = r_eff * (Dll ** 2) * Dpp

            lap_Wg = laplacian(Wg, rho_wg)
            lap_Dpp = laplacian(Dpp)

            if alpha_ramp_h > 0.0:
                frac = 0.5 * (1.0 + jnp.tanh((t - T_late_sec) / ramp_sec))
            else:
                frac = jnp.where(t >= T_late_sec, 1.0, 0.0)
            current_alpha = alpha + frac * (alpha_late - alpha)

            dWg = turing_reaction - (k1 * Wg) + (D1 * lap_Wg)
            dDpp = current_alpha - turing_reaction - (k2 * Dpp) + (D2 * lap_Dpp)

            # 4. Fz4 dynamics (Wg repression keeps the HCR "hole")
            if Fz4_knockout:
                Fz4_next = jnp.zeros((ny, nx))
            else:
                wg_repression = 1.0 / (1.0 + (Wg / Kwg) ** n_wg)
                dFz4 = (beta_f * Y_gradient * wg_repression) - (k_f * Fz4)
                Fz4_next = Fz4 + dt * dFz4
                Fz4_next = jnp.where(vein_mask | mask_Fz4, 0.0, Fz4_next)
                Fz4_next = Fz4_next.at[0, :].set(0.0)

            # 5. Downstream readouts
            upstream_driver_s = Dll ** 3
            prod_s = (upstream_driver_s ** n_hill) / (
                (Ks ** n_hill) + (upstream_driver_s ** n_hill) + 1e-8)
            feedback_antp = (Antp ** n_hill) / (
                (Kas ** n_hill) + (Antp ** n_hill) + 1e-8)
            dS = (beta_s * prod_s * (1.0 + feedback_antp)) - (k_s * S)

            upstream_driver_a = Dll * S
            prod_a = (upstream_driver_a ** n_hill) / (
                (Ka ** n_hill) + (upstream_driver_a ** n_hill) + 1e-8)
            repressor_factor = 1.0 / (1.0 + (X_repressor / K_x) ** n_rep)
            dAntp = (beta_a * prod_a * repressor_factor) - (k_a * Antp)

            Wg_next = Wg + dt * dWg
            Dpp_next = Dpp + dt * dDpp
            S_next = S + dt * dS
            Antp_next = Antp + dt * dAntp

            # 6. Clone masks and vein sinks
            Wg_next = jnp.where(vein_mask | mask_Wg, 0.0, Wg_next)
            Dpp_next = jnp.where(vein_mask | mask_Dpp, 0.0, Dpp_next)
            S_next = jnp.where(vein_mask | mask_S, 0.0, S_next)
            Antp_next = jnp.where(vein_mask | mask_Antp, 0.0, Antp_next)

            # 7. Wing margin (row 0). This overrides the vein sink at the two
            #    corner cells where margin meets vein.
            Wg_next = Wg_next.at[0, :].set(cmargin)
            Dpp_next = Dpp_next.at[0, :].set(Dpp_next[1, :])
            S_next = S_next.at[0, :].set(S_next[1, :])
            Antp_next = Antp_next.at[0, :].set(0.0)

            return (Wg_next, Dpp_next, S_next, Antp_next, Fz4_next), None

        return jax.lax.scan(step_fn, state, times)[0]

    Wg = jnp.zeros((ny, nx)).at[0, :].set(cmargin)
    Dpp = jnp.zeros((ny, nx))          # Connahs et al.: no substrate at t = 0
    S = jnp.zeros((ny, nx))
    Antp = jnp.zeros((ny, nx))
    Fz4 = jnp.zeros((ny, nx)) if Fz4_knockout else (beta_f * Y_gradient / k_f)

    state = (Wg, Dpp, S, Antp, Fz4)
    current_time_sec = 0.0

    pheno_title = f"{'Fz4 KO' if Fz4_knockout else 'WT'} | "
    # width only applies to center and corner clones
    width_tag = f" w={clone_width}" if uses_width else ""
    pheno_title += (f"{', '.join(target_genes)} {clone_shape.capitalize()} "
                    f"Clone{width_tag}"
                    if clone_shape != 'none' else "No Clone")
    pheno_title += f" | r_max={r_max:.2e}"

    params = dict(r_max=r_max, D1=D1, D2=D2, rho_wg=rho_wg, k1=k1, k2=k2,
                  alpha=alpha,
                  alpha_late_mult=alpha_late_mult, T_late_days=T_late_days,
                  alpha_ramp_h=alpha_ramp_h, k1l=k1l, k3=k3, f_s=f_s, K_sd=K_sd, n_sd=n_sd, beta_f=beta_f,
                  k_f=k_f, Kwg=Kwg, Kfz=Kfz, n_gate=n_gate, n_wg=n_wg,
                  beta_s=beta_s,
                  beta_a=beta_a, k_s=k_s, k_a=k_a, Ks=Ks, Ka=Ka, Kas=Kas,
                  n_hill=n_hill, X0=X0, lambda_X=lambda_X, K_x=K_x,
                  n_rep=n_rep, dx=dx, dt=dt, nx=nx, ny=ny, Lx=Lx, Ly=Ly,
                  cmargin=cmargin, total_days=total_days,
                  clone_shape=clone_shape,
                  clone_width=clone_width if uses_width else None,
                  target_genes=target_genes, Fz4_knockout=Fz4_knockout)

    # ---------- filename stem ----------
    genotype_prefix = "Fz4_KO_global" if Fz4_knockout else "WT"
    if clone_shape == 'none':
        clone_suffix = "no_clone"
    else:
        targets = "_".join(target_genes)
        width_sfx = f"_w{clone_width}" if uses_width else ""
        clone_suffix = f"{targets}_clone_{clone_shape}{width_sfx}"
    stem = f"{genotype_prefix}_{clone_suffix}"
    stem = stem.replace(' ', '_').replace(',', '').replace('=', '')

    # ---------- output folder, next to app.py ----------
    try:
        APP_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:                      # interactive / notebook fallback
        APP_DIR = os.getcwd()
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(APP_DIR, "simulation_output", f"{stamp}_{stem}")
    frames_dir = os.path.join(run_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    st.write(f"### Running Simulation: {pheno_title}")
    st.caption(f"Saving to {run_dir}")
    progress_bar = st.progress(0)
    status_text = st.empty()
    gif_placeholder = st.empty()

    tmp_dir = frames_dir            # frames are kept, not thrown away
    if True:
        frame_files = []
        clone_mask_cpu = np.array(clone_mask_jnp)
        ext = [0, Lx, 0, Ly]
        y_plot_vals = np.linspace(0, Ly, ny)

        for frame in range(num_frames + 1):
            day = current_time_sec / (24 * 3600)
            Wg_curr, Dpp_curr, S_curr, Antp_curr, Fz4_curr = state
            S_np = np.array(S_curr)
            Dll_curr = (k1l / k3) * np.array(Wg_curr) * (
                1.0 + f_s * S_np ** n_sd / (K_sd ** n_sd + S_np ** n_sd + 1e-8))
            Dll_curr = np.where(np.array(mask_Dll), 0.0, Dll_curr)
            r_eff_curr = np.array(r_max / (1.0 + (Fz4_curr / Kfz) ** n_gate))

            fig = plt.figure(figsize=(17, 9))
            gs = plt.GridSpec(2, 4, figure=fig)
            fig.suptitle(f"Eyespot Simulation ({pheno_title}) - Day {day:.2f}",
                         fontsize=13, fontweight='bold')

            axes = []

            ax0 = fig.add_subplot(gs[0, 0]); axes.append(ax0)
            im0 = ax0.imshow(Wg_curr, origin='lower', extent=ext,
                             cmap='viridis', vmin=0, vmax=120)
            ax0.set_title("Wg Module"); fig.colorbar(im0, ax=ax0)

            ax1 = fig.add_subplot(gs[0, 1]); axes.append(ax1)
            im1 = ax1.imshow(Dpp_curr, origin='lower', extent=ext,
                             cmap='plasma', vmin=0, vmax=40)
            ax1.set_title("Dpp Module"); fig.colorbar(im1, ax=ax1)

            ax2 = fig.add_subplot(gs[0, 2]); axes.append(ax2)
            im2 = ax2.imshow(Dll_curr, origin='lower', extent=ext, cmap='hot',
                             vmin=0, vmax=120)
            ax2.set_title("Distal-less (Dll)"); fig.colorbar(im2, ax=ax2)

            ax3 = fig.add_subplot(gs[0, 3]); axes.append(ax3)
            im3 = ax3.imshow(Fz4_curr, origin='lower', extent=ext, cmap='bone',
                             vmin=0, vmax=800)
            ax3.set_title("frizzled4 (Fz4)"); fig.colorbar(im3, ax=ax3)

            ax4 = fig.add_subplot(gs[1, 0]); axes.append(ax4)
            im4 = ax4.imshow(S_curr, origin='lower', extent=ext,
                             cmap='inferno', vmin=0, vmax=2600)
            ax4.set_title("Spalt (S)"); fig.colorbar(im4, ax=ax4)

            ax5 = fig.add_subplot(gs[1, 1]); axes.append(ax5)
            im5 = ax5.imshow(Antp_curr, origin='lower', extent=ext,
                             cmap='magma', vmin=0, vmax=2600)
            ax5.set_title("Antennapedia (Antp)"); fig.colorbar(im5, ax=ax5)

            ax7 = fig.add_subplot(gs[1, 2]); axes.append(ax7)
            im7 = ax7.imshow(r_eff_curr, origin='lower', extent=ext,
                             cmap='cividis', vmin=0, vmax=float(r_max))
            ax7.set_title("Local coupling $r_{eff}$")
            fig.colorbar(im7, ax=ax7)

            if clone_shape != 'none':
                for ax in axes:
                    ax.contour(clone_mask_cpu, levels=[0.5], colors='cyan',
                               linewidths=1.5, alpha=0.8, extent=ext)

            # ---- profiles, sampled outside the clone
            ax6 = fig.add_subplot(gs[1, 3])
            in_clone = clone_mask_cpu.any(axis=0)
            S_cpu = np.array(S_curr)
            if clone_shape == 'none' or not in_clone.any():
                sample_idx = nx // 2
                title_suffix = ""
            else:
                col_peak = np.max(S_cpu, axis=0).astype(float)
                col_peak[in_clone] = -np.inf      # never sample the clone
                sample_idx = int(np.argmax(col_peak))
                title_suffix = f" (x={sample_idx})"

            ax6.plot(Dll_curr[:, sample_idx], y_plot_vals, 'r-', linewidth=2,
                     label="Dll")
            ax6.plot(S_cpu[:, sample_idx], y_plot_vals, 'm-', linewidth=2,
                     label="Spalt")
            ax6.plot(np.array(Antp_curr)[:, sample_idx], y_plot_vals,
                     'orange', linewidth=2, label="Antp")
            ax6.plot(np.array(Fz4_curr)[:, sample_idx], y_plot_vals, 'c--',
                     linewidth=2, label="Fz4")
            ax6.set_title(f"Profiles{title_suffix}")
            ax6.set_ylim(0, Ly)
            ax6.grid(True, alpha=0.3)
            ax6.legend(loc='upper right', fontsize=8)

            plt.tight_layout()

            frame_path = os.path.join(tmp_dir, f"frame_{frame:04d}.png")
            plt.savefig(frame_path, dpi=110)
            plt.close()
            frame_files.append(frame_path)

            progress_bar.progress(int((frame / num_frames) * 100))
            status_text.text(f"Computing Day {day:.2f}...")

            if frame < num_frames:
                state = update_n_steps(state, current_time_sec)
                current_time_sec += steps_per_frame * dt

        status_text.text("Compiling GIF Animation...")

        images = [Image.open(f) for f in frame_files]
        final_gif_path = os.path.join(run_dir, f"{stem}.gif")
        images[0].save(final_gif_path, save_all=True,
                       append_images=images[1:], duration=100, loop=0)
        with open(os.path.join(run_dir, f"params_{stem}.json"), "w") as fh:
            json.dump(params, fh, indent=2)

        status_text.text("Packaging Frames into ZIP...")
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
            for frame_file in frame_files:
                zip_file.write(frame_file, os.path.basename(frame_file))
        zip_buffer.seek(0)

        status_text.text("Simulation Complete.")
        progress_bar.empty()

        with open(final_gif_path, "rb") as fh:
            gif_bytes = fh.read()

        st.session_state["sim"] = dict(
            gif_bytes=gif_bytes,
            zip_bytes=zip_buffer.getvalue(),
            params_json=json.dumps(params, indent=2),
            stem=stem,
            title=pheno_title,
            run_dir=run_dir,
            n_frames=len(frame_files),
        )

# =====================================================================
# RESULTS  -  rendered from session_state so a download click (which
# reruns the script) does not wipe the animation off the page.
# =====================================================================
sim = st.session_state.get("sim")
if sim:
    st.markdown("---")
    st.write(f"### Result: {sim['title']}")
    data_url = base64.b64encode(sim["gif_bytes"]).decode("utf-8")
    st.markdown(
        f'<img src="data:image/gif;base64,{data_url}" '
        f'alt="Eyespot Simulation Animation" width="100%">',
        unsafe_allow_html=True,
    )
    st.caption(
        f"Saved to {sim['run_dir']}  \u00b7  "
        f"{sim['n_frames']} frames in frames/  \u00b7  "
        f"{sim['stem']}.gif  \u00b7  params_{sim['stem']}.json"
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button("\U0001F4E5 Download Simulation GIF",
                           data=sim["gif_bytes"],
                           file_name=f"{sim['stem']}.gif",
                           mime="image/gif", key="dl_gif")
    with c2:
        st.download_button("\U0001F4C4 Download Parameters (JSON)",
                           data=sim["params_json"],
                           file_name=f"params_{sim['stem']}.json",
                           mime="application/json", key="dl_json")
    with c3:
        st.download_button("\U0001F5C2\uFE0F Download All Frames (ZIP)",
                           data=sim["zip_bytes"],
                           file_name=f"{sim['stem']}_frames.zip",
                           mime="application/zip", key="dl_zip")
