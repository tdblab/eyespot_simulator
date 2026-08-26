import streamlit as st
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import os
import tempfile
import base64
from PIL import Image

# ==========================================
# PAGE CONFIGURATION & UI HEADER
# ==========================================
st.set_page_config(page_title="Eyespot Simulator", page_icon="🦋", layout="wide")

st.title("🦋 Bicyclus anynana Eyespot Simulator")
st.markdown("""
Welcome to the interactive *in silico* CRISPR clone engine. 
Explore how somatic mutations affect morphogen reaction-diffusion and Turing patterning.
**[📄 Read the Mathematical Documentation & Network Architecture (PDF)](#)**
""")

# ==========================================
# SIDEBAR: RUN BUTTON AT THE VERY TOP
# ==========================================
with st.sidebar:
    # RUN BUTTON PLACED AT THE VERY TOP
    run_button = st.button("🚀 Run Simulation", type="primary", use_container_width=True)
    
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
    
    clone_width = st.slider("Clone Width (Pixels)", min_value=2, max_value=40, value=2, step=2)
    Fz4_knockout = st.checkbox("Global Fz4 Knockout (Whole Tissue)", value=False)
    
    st.markdown("---")
    st.header("⚙️ Coupling & Network Parameters")
    
    # Coupling Strengths
    K_wt = st.number_input("Wild-Type (K_wt)", value=1.4e-7, format="%.2e")
    K_split = st.number_input("Global Fz4 KO (K_split)", value=1.72e-7, format="%.2e")
    K_core_ko = st.slider("Core KO Coupling", min_value=1.50e-7, max_value=2.00e-7, value=1.90e-7, step=0.01e-7, format="%.2e")
    
    # ==========================================
    # DYNAMIC STATE ROUTER LOGIC
    # ==========================================
    core_genes_selected = any(gene in target_genes for gene in ['Wg', 'Dpp', 'Dll'])
    has_clone = clone_shape != 'none'
    
    if Fz4_knockout:
        K_eff = K_split
        k_source = "Global Fz4 KO (K_split)"
    elif has_clone and core_genes_selected:
        K_eff = K_core_ko
        k_source = "Core Gene Clone (K_core_ko)"
    else:
        K_eff = K_wt
        k_source = "Wild-Type (K_wt)"
        
    st.info(f"**Active Coupling:** {K_eff:.2e}\n*(Source: {k_source})*")
    
    st.markdown("---")
    
    # MODULE 1: Core Morphogens
    with st.expander("1. Core Morphogens (Wg & Dpp)", expanded=False):
        D1 = st.number_input("Wg Diffusion (D1)", value=0.01, format="%.4f")
        D2 = st.number_input("Dpp Diffusion (D2)", value=0.12, format="%.4f")
        k1 = st.number_input("Wg Degradation (k1)", value=0.1e-3, format="%.2e")
        k2 = st.number_input("Dpp Degradation (k2)", value=0.08e-3, format="%.2e")
        alpha = st.number_input("Dpp Production (alpha)", value=6.2e-3, format="%.2e")
        alpha_late_mult = st.slider("Maturation Drop (alpha_late %)", 0.50, 1.00, 0.80, 0.01)
        T_late_days = st.number_input("Maturation Time (Days)", value=2.0, step=0.1)

    # MODULE 2: Receptors & Transducers
    with st.expander("2. Receptors (Fz4) & Transducers (Dll)", expanded=False):
        k1l = st.number_input("Dll Activation (k1l)", value=1.0e-3, format="%.2e")
        k3 = st.number_input("Dll Degradation (k3)", value=1.0e-3, format="%.2e")
        beta_f = st.number_input("Fz4 Production (beta_f)", value=1.2, step=0.1)
        k_f = st.number_input("Fz4 Degradation (k_f)", value=1.0e-3, format="%.2e")
        Kwg = st.number_input("Wg Repression of Fz4 (Kwg)", value=30.0, step=1.0)
        Kfz = st.number_input("Fz4 Inhibition of Dll (Kfz)", value=1500.0, step=100.0)
        n_fz = st.slider("Fz4 Hill Coefficient (n_fz)", 1.0, 5.0, 2.0, 0.1)

    # MODULE 3: Downstream Network
    with st.expander("3. Downstream TFs (Spalt & Antp)", expanded=False):
        beta_s = st.number_input("Spalt Production (beta_s)", value=2.8, step=0.1)
        beta_a = st.number_input("Antp Production (beta_a)", value=1.8, step=0.1)
        k_s = st.number_input("Spalt Degradation (k_s)", value=1.0e-3, format="%.2e")
        k_a = st.number_input("Antp Degradation (k_a)", value=2.0e-3, format="%.2e")
        Ks = st.number_input("Spalt Activation Threshold (Ks)", value=5.0e5, format="%.2e")
        Ka = st.number_input("Antp Activation Threshold (Ka)", value=1.0e4, format="%.2e")
        Kas = st.number_input("Antp Feedback Threshold (Kas)", value=200.0, step=10.0)
        n_hill = st.slider("TF Hill Coefficient (n_hill)", 1.0, 5.0, 3.0, 0.1)

# ==========================================
# CORE JAX JIT FUNCTIONS 
# ==========================================
dx = 2.5       
dt = 1.8       
nx, ny = 60, 105 
Lx, Ly = 150, 262 
cmargin = 65.0 

@jax.jit
def laplacian(Z):
    Z_top = jnp.roll(Z, shift=-1, axis=0)
    Z_bottom = jnp.roll(Z, shift=1, axis=0)
    Z_left = jnp.roll(Z, shift=-1, axis=1)
    Z_right = jnp.roll(Z, shift=1, axis=1)
    return (Z_top + Z_bottom + Z_left + Z_right - 4.0 * Z) / (dx**2)

def build_clone_mask(shape, width):
    mask = np.zeros((ny, nx), dtype=bool)
    if shape == 'center':
        cx = nx // 2
        mask[:, cx - (width//2) : cx + (width//2)] = True
    elif shape == 'diagonal':
        for i in range(ny):
            for j in range(nx):
                if i > ny - (j * (ny/nx)) - (ny//4): 
                    mask[i, j] = True
    elif shape == 'corner':
        mask[ny - (width*2):, :width*2] = True
    elif shape == 'full':
        mask[:, :] = True
    return jnp.array(mask)

# ==========================================
# MAIN SIMULATION EXECUTION
# ==========================================
if run_button:
    r2 = K_eff * (k3**2) / (k1l**2)
    r1 = r2

    alpha_late = alpha_late_mult * alpha
    T_late_sec = T_late_days * 24 * 3600  

    total_days = 6.0
    total_steps = int((total_days * 24 * 3600) / dt)
    num_frames = 60 
    steps_per_frame = total_steps // num_frames

    y_vals = jnp.linspace(0, Ly, ny)
    x_vals = jnp.linspace(0, Lx, nx)
    Y, X_grid = jnp.meshgrid(y_vals, x_vals, indexing='ij')

    Y_gradient = Y / Ly  
    X_repressor = 10.0 * jnp.exp(-Y / 40.0)
    K_x, n_repressor_hill = 2.0, 2.0

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
            
            Dll = (k1l / k3) * Wg
            Dll = jnp.where(mask_Dll, 0.0, Dll)
            
            current_alpha = jnp.where(t >= T_late_sec, alpha_late, alpha)
            
            lap_Wg = laplacian(Wg)
            lap_Dpp = laplacian(Dpp)

            active_fz4 = jnp.where(Fz4_knockout, 0.0, Fz4)
            fz4_repression = 1.0 / (1.0 + (active_fz4 / Kfz)**n_fz)
            dll_sq_regulated = (Dll**2) * fz4_repression
            
            dWg = (r1 * dll_sq_regulated * Dpp) - (k1 * Wg) + (D1 * lap_Wg)
            dDpp = current_alpha - (r2 * (Dll**2) * Dpp) - (k2 * Dpp) + (D2 * lap_Dpp)
            
            if Fz4_knockout:
                Fz4_next = jnp.zeros((ny, nx))
            else:
                wg_repression = 1.0 / (1.0 + (Wg / Kwg)**n_fz)
                dFz4 = (beta_f * Y_gradient * wg_repression) - (k_f * Fz4)
                Fz4_next = Fz4 + dt * dFz4
                Fz4_next = jnp.where(vein_mask | mask_Fz4, 0.0, Fz4_next)
                Fz4_next = Fz4_next.at[0, :].set(0.0)

            upstream_driver_s = (Dll**2) * Wg
            prod_s = (upstream_driver_s**n_hill) / ((Ks**n_hill) + (upstream_driver_s**n_hill) + 1e-8)
            feedback_antp = (Antp**n_hill) / ((Kas**n_hill) + (Antp**n_hill) + 1e-8)
            dS = (beta_s * prod_s * (1.0 + feedback_antp)) - (k_s * S)
            
            upstream_driver_a = Dll * S
            prod_a = (upstream_driver_a**n_hill) / ((Ka**n_hill) + (upstream_driver_a**n_hill) + 1e-8)
            repressor_factor = 1.0 / (1.0 + (X_repressor / K_x)**n_repressor_hill)
            dAntp = (beta_a * prod_a * repressor_factor) - (k_a * Antp)

            Wg_next = Wg + dt * dWg
            Dpp_next = Dpp + dt * dDpp
            S_next = S + dt * dS
            Antp_next = Antp + dt * dAntp

            Wg_next = jnp.where(vein_mask | mask_Wg, 0.0, Wg_next)
            Dpp_next = jnp.where(vein_mask | mask_Dpp, 0.0, Dpp_next)
            S_next = jnp.where(vein_mask | mask_S, 0.0, S_next)
            Antp_next = jnp.where(vein_mask | mask_Antp, 0.0, Antp_next)

            Wg_next = Wg_next.at[0, :].set(cmargin)          
            Dpp_next = Dpp_next.at[0, :].set(Dpp_next[1, :]) 
            S_next = S_next.at[0, :].set(S_next[1, :])
            Antp_next = Antp_next.at[0, :].set(0.0) 

            return (Wg_next, Dpp_next, S_next, Antp_next, Fz4_next), None
            
        return jax.lax.scan(step_fn, state, times)[0]

    Wg = jnp.zeros((ny, nx)).at[0, :].set(cmargin)
    Dpp = jnp.full((ny, nx), alpha / k2)
    S = jnp.zeros((ny, nx))
    Antp = jnp.zeros((ny, nx))
    Fz4 = jnp.zeros((ny, nx)) if Fz4_knockout else (beta_f * Y_gradient / k_f)
    
    state = (Wg, Dpp, S, Antp, Fz4)
    current_time_sec = 0.0

    pheno_title = f"{'Fz4 KO' if Fz4_knockout else 'WT'} | "
    pheno_title += f"{', '.join(target_genes)} {clone_shape.capitalize()} Clone" if clone_shape != 'none' else "No Clone"

    st.write(f"### Running Simulation: {pheno_title}")
    progress_bar = st.progress(0)
    status_text = st.empty()
    gif_placeholder = st.empty()

    with tempfile.TemporaryDirectory() as tmp_dir:
        frame_files = []
        clone_mask_cpu = np.array(clone_mask_jnp)
        ext = [0, Lx, 0, Ly]
        y_plot_vals = np.linspace(0, Ly, ny)
        
        for frame in range(num_frames + 1):
            day = current_time_sec / (24 * 3600)
            Wg_curr, Dpp_curr, S_curr, Antp_curr, Fz4_curr = state
            Dll_curr = (k1l / k3) * Wg_curr
            Dll_curr = np.where(np.array(mask_Dll), 0.0, Dll_curr)  
            
            fig = plt.figure(figsize=(16, 10))
            gs = plt.GridSpec(2, 4, figure=fig)
            fig.suptitle(f"Eyespot Simulation ({pheno_title}) - Day {day:.2f}", fontsize=14, fontweight='bold')
            
            axes = []
            
            ax0 = fig.add_subplot(gs[0, 0]); axes.append(ax0)
            im0 = ax0.imshow(Wg_curr, origin='lower', extent=ext, cmap='viridis', vmin=0, vmax=80)
            ax0.set_title("Wg Module"); fig.colorbar(im0, ax=ax0)
            
            ax1 = fig.add_subplot(gs[0, 1]); axes.append(ax1)
            im1 = ax1.imshow(Dpp_curr, origin='lower', extent=ext, cmap='plasma', vmin=0, vmax=40)
            ax1.set_title("Dpp Module"); fig.colorbar(im1, ax=ax1)

            ax2 = fig.add_subplot(gs[0, 2]); axes.append(ax2)
            im2 = ax2.imshow(Dll_curr, origin='lower', extent=ext, cmap='hot', vmin=0, vmax=100)
            ax2.set_title("Distal-less (Dll)"); fig.colorbar(im2, ax=ax2)

            ax3 = fig.add_subplot(gs[0, 3]); axes.append(ax3)
            im3 = ax3.imshow(Fz4_curr, origin='lower', extent=ext, cmap='bone', vmin=0, vmax=600)
            ax3.set_title("frizzled4 (Fz4)"); fig.colorbar(im3, ax=ax3)
            
            ax4 = fig.add_subplot(gs[1, 0]); axes.append(ax4)
            im4 = ax4.imshow(S_curr, origin='lower', extent=ext, cmap='inferno', vmin=0, vmax=600)
            ax4.set_title("Spalt (S)"); fig.colorbar(im4, ax=ax4)

            ax5 = fig.add_subplot(gs[1, 1]); axes.append(ax5)
            im5 = ax5.imshow(Antp_curr, origin='lower', extent=ext, cmap='magma', vmin=0, vmax=600)
            ax5.set_title("Antennapedia (Antp)"); fig.colorbar(im5, ax=ax5)
            
            if clone_shape != 'none':
                for ax in axes:
                    ax.contour(clone_mask_cpu, levels=[0.5], colors='cyan', linewidths=1.5, alpha=0.8, extent=ext)

            ax6 = fig.add_subplot(gs[1, 2:])
            center_idx = nx // 2
            
            if clone_shape == 'center':
                left_hemisphere_S = S_curr[:, :center_idx - clone_width]
                sample_idx = int(np.argmax(np.max(left_hemisphere_S, axis=0)))
                title_suffix = f" (Sampled at peak x={sample_idx})"
            else:
                sample_idx = center_idx
                title_suffix = ""

            ax6.plot(Dll_curr[:, sample_idx], y_plot_vals, 'r-', linewidth=2, label="Dll")
            ax6.plot(S_curr[:, sample_idx], y_plot_vals, 'm-', linewidth=2, label="Spalt")
            ax6.plot(Antp_curr[:, sample_idx], y_plot_vals, 'orange', linewidth=2, label="Antp")
            ax6.plot(Fz4_curr[:, sample_idx], y_plot_vals, 'c--', linewidth=2, label="Fz4")
            
            ax6.set_title(f"Centerline Profiles{title_suffix}")
            ax6.set_ylim(0, Ly)
            ax6.grid(True, alpha=0.3)
            ax6.legend(loc='upper right')

            plt.tight_layout()
            
            frame_path = os.path.join(tmp_dir, f"frame_{frame:04d}.png")
            plt.savefig(frame_path, dpi=120)
            plt.close()
            frame_files.append(frame_path)
            
            progress = int((frame / num_frames) * 100)
            progress_bar.progress(progress)
            status_text.text(f"Computing Day {day:.2f}...")

            if frame < num_frames:
                state = update_n_steps(state, current_time_sec)
                current_time_sec += steps_per_frame * dt

        status_text.text("Compiling GIF Animation...")
        
        images = [Image.open(f) for f in frame_files]
        final_gif_path = "simulation_output.gif"
        images[0].save(final_gif_path, save_all=True, append_images=images[1:], duration=100, loop=0)
        
        status_text.text("Simulation Complete!")
        progress_bar.empty()
        
        # RENDER THE ANIMATED GIF VIA BASE64 HTML
        with open(final_gif_path, "rb") as file:
            contents = file.read()
            data_url = base64.b64encode(contents).decode("utf-8")
            gif_placeholder.markdown(
                f'<img src="data:image/gif;base64,{data_url}" alt="Eyespot Simulation Animation" width="100%">',
                unsafe_allow_html=True
            )
        
        with open(final_gif_path, "rb") as file:
            st.download_button(
                label="📥 Download Simulation GIF",
                data=file,
                file_name=f"eyespot_{pheno_title.replace(' | ', '_').replace(' ', '_').replace(',', '')}.gif",
                mime="image/gif"
            )
