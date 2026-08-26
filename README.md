# 🦋 Eyespot Simulator

An interactive *in silico* CRISPR clone engine and reaction-diffusion simulator modeling butterfly wing eyespot development (*Bicyclus anynana*). 

This tool bridges mathematical modeling (JAX-accelerated PDE solvers) with modern EvoDevo biology, allowing researchers to simulate cell-autonomous and non-cell-autonomous somatic mosaic knockouts within a multi-gene regulatory network.

---

## 🧬 Network Architecture & Mathematical Model

The simulator models a 5-gene regulatory network governing eyespot formation across the pupal wing sector:
* **Core Turing Loop:** Wingless (`Wg`) and Decapentaplegic (`Dpp`) interacting via non-linear reaction-diffusion kinetics.
* **Transducer Module:** Distal-less (`Dll`), slaved to Wg dynamics.
* **Receptor Feedback:** Frizzled4 (`Fz4`), providing dynamic spatial patterning control.
* **Downstream Target TFs:** Spalt (`S`) and Antennapedia (`Antp`), executing color-pattern readout cell-autonomously.

📄 **[Read the Full Mathematical Documentation & Network Architecture (PDF)](#)** *(Link your PDF here)*

<img width="749" height="681" alt="Screenshot 2026-08-27 064916" src="https://github.com/user-attachments/assets/62b4bfaf-10d2-4e5f-b7fb-ebafe3f765e2" />


---

## 🚀 Features

* **Interactive CRISPR Clone Engine:** Select from preset somatic clone topologies (`Center`, `Diagonal`, `Corner`, `Full`) or custom configurations.
* **Multi-Gene Knockouts:** Independently or simultaneously knock out `Wg`, `Dpp`, `Dll`, `Fz4`, `Spalt`, or `Antp` from time $t = 0$.
* **Full Parameter Control:** Tune diffusion coefficients ($D_1, D_2$), degradation rates, Hill coefficients, coupling strengths ($K_{wt}, K_{split}, K_{core\_ko}$), and temporal maturation drops ($\alpha_{late}$).
* **Real-Time Visualization & Export:** Renders 2D spatial distribution heatmaps alongside real-time centerline profile traces, compiled into a smooth browser-rendered animation GIF with one-click download options.

---

## 🛠️ Step-by-Step Installation Guide (For Biologists & Researchers)

To ensure a clean installation that doesn't conflict with your system's existing Python packages, we recommend using **Miniconda** or **Anaconda**. 

### Step 1: Install Miniconda (If not already installed)
Download and install Miniconda for your operating system (Windows, macOS, or Linux) from the [official Miniconda website](https://docs.conda.io/en/latest/miniconda.html).

### Step 2: Open your Terminal / Anaconda Prompt
* **Windows:** Open **Anaconda Prompt** from your start menu.
* **macOS / Linux:** Open your standard **Terminal** app.

### Step 3: Create and Activate a Dedicated Virtual Environment
Create a clean Python environment named `eyespot_env` by running:
```bash
conda create -n eyespot_env python=3.10 -y
```

**Activate the environment**
```bash
conda activate eyespot_env
```

### Step 5: Install Dependencies
Install all required scientific and web framework packages inside your active environment:
```bash
pip install streamlit jax jaxlib matplotlib pillow numpy
```

### Step 6: Launch the Eyespot Simulator
Run the Streamlit application:
```bash
streamlit run app.py
```
### Run the app using the following link
https://eyespotsimulator-kzwxkir9m7zjwyjrtczruw.streamlit.app/
