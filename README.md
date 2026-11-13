# VaM_Physical_AI_from_Mimickit

Run physical AI trained with MimicKit inside Virt-A-Mate (VaM). Interact with the AI character using VR controllers (push and grab), control its following and facing behavior, and enable furniture obstacle avoidance.

> 💡 **Fork Notice:** This repository is a fork of the official [MimicKit](https://github.com/xbpeng/MimicKit) framework, combining core motion imitation with custom environment files and plugins for real-time UDP inference in Virt-A-Mate.

> ⚠️ **Disclaimer:** This is an unofficial community project. It is not affiliated with, associated with, or endorsed by MimicKit or MeshedVR.

📺 **DEMO:** [Watch on YouTube](https://www.youtube.com/watch?v=8Xqi1D7QRnU)

---

## Overview

This project runs AI inference via **Newton** while Virt-A-Mate runs. **A pre-trained physical AI model is included for immediate use.** The AI character can follow the player, face a direction selected with a VR controller, and detect furniture as obstacles to avoid. **Furniture collision support enables physical interactions with obstacles, including collision responses, balance recovery, and falling. Furniture manipulation is not currently supported.**

- **Desktop Mode:** Runs on a single PC. You can test the AI's **follow feature** without a VR headset, although VR-based physical interactions are unavailable.
- **VR Mode:** Requires substantial computing power—typically a separate PC with an NVIDIA GPU for AI inference alongside the VaM PC. *(An RTX 4070 Ti single-PC setup proved insufficient during testing.)*
  - **Development & Tested VR Setup (Dual-PC):**
    - **VaM PC:** NVIDIA RTX 4070 Ti / Intel Core i5-13600KF
    - **Newton AI Inference PC:** NVIDIA RTX 3090 / Intel Core i5-11400F
  - This dual-PC configuration was used for development, VR testing, and gameplay.
- **Hardware Compatibility:** Tested with **Meta Quest 3 (OpenXR)**. Other HMD/controller tracking configurations have not been verified.

---

## Network & Components

Communication between VaM and Newton uses UDP:

```text
Virt-A-Mate (VaM)                          Newton / MimicKit
  VR controllers + HMD  ── UDP 9999 ──────► VR tracking / interaction
  Furniture transforms ── UDP 9001 ──────► Position and rotation updates
  Furniture shapes     ── UDP 9002 ──────► Collider shape registration
  Character bones      ◄─ UDP 9998 ─────── AI motion output
```

- IP addresses and ports can be configured for single-PC or dual-PC setups. For a single-PC setup, use `127.0.0.1` as the local IP address; for two PCs, configure the relevant LAN IP addresses.
- **VaM side:** Configure VR hand tracking output, furniture synchronization, and character bone reception through the corresponding plugin UIs.
- **Newton/MimicKit side:** Adjust the relevant UDP settings in `data/envs/amp_g1_vam_env.yaml` (including `vam_udp_ip`, `vam_udp_port`, and `vrhand_udp_port`).
- **VaM plugins:**
  - `Receiver_mimickit_public.cs` — **Person plugin** for receiving AI character motion.
  - `VRhand_mimickit_public.cs` — **Scene plugin** for VR controller/HMD data and controls.
  - `Atom_sync_mimickit_public.cs` — **Scene plugin** for furniture Atom transform and collider shape synchronization.

### Furniture Synchronization

`Atom_sync_mimickit_public.cs` sends furniture Atom data from VaM to Newton:

- **UDP 9001:** Furniture Atom position and rotation updates during operation.
- **UDP 9002:** Furniture Atom collider shape data, sent when the plugin starts and when new furniture Atoms are added to the scene.
- **Position and rotation changes are synchronized**, but **scale changes are not synchronized**. If you change a furniture Atom's scale, you must **re-register its collider shape in Newton** and restart Newton to use the updated shape.

### Registering Furniture in Newton (First Use)

Each new furniture object must be registered before use. Registration is automatic, but **Newton must be restarted after registration** before the collider is available for normal use.

1. Start **Newton** and **Virt-A-Mate**.
2. Make sure the AI character is running and its motion is synchronized with the character in VaM.
3. Add the furniture Atom you want to use to the VaM scene.
4. Newton automatically processes the received collider shapes, simplifies them, and saves the furniture collider assets.
5. **Restart Newton** to complete the furniture setup.

Registered furniture can be reused in later sessions without repeating this process, provided its scale has not changed. Saved collider assets are located in:

```text
data/assets/objects/vam_env_boxes/
```

**Note:** If you change the scale of an already-registered furniture Atom, its stored collider geometry must be re-registered. Moving or rotating the Atom alone does not require re-registration.

---

## VR Controls

Use a **double trigger press (×2)** on the corresponding VR controller. Perform these controls when not interacting with or grabbing an object.

| Controller | Action | Result |
| --- | --- | --- |
| **Left — double trigger** | Point at empty space and double press | Toggle AI target following **ON / OFF** (**Follow / Stay**). |
| **Right — double trigger** | Point in the desired direction and double press | Set the AI's facing direction to the direction indicated by the controller. |
| **Right — double trigger** | Point at the ground and double press | Return to having the AI **look toward you**. |

You can also **push and grab the AI character** with your VR controllers.

---

## Setup Instructions

### Prerequisites

1. **Git:** Install from [git-scm.com](https://git-scm.com).
2. **Miniconda / Anaconda:** Install from [anaconda.com](https://anaconda.com).

### Installation & Execution

Run the following commands in your **Anaconda Prompt**.

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yuyaslabo/MimicKit_VaM
   cd MimicKit_VaM
   ```

2. **Download the official MimicKit assets and motion data:**

   After cloning this repository, visit the [official MimicKit installation instructions](https://github.com/xbpeng/MimicKit#installation).

   In **Step 3**, follow the download link for the assets and motion data (`.zip` archive).

   Extract the archive and place its contents inside the `data/` directory of this cloned repository. Preserve the original directory structure so the assets and motion files are available under `data/` (e.g., `data/motions/`).

3. **Create and activate the Python environment:**

   ```bash
   conda create --name mimickit python=3.11 -y
   conda activate mimickit
   ```

4. **Install Newton Simulator Backend (v1.0.0):**

   ```bash
   pip install newton==1.0.0
   ```

5. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   > **Troubleshooting (`J_ki` Error):** Depending on your GPU architecture (e.g., RTX 3090), a `J_ki`-related Warp compilation error may occur. If you encounter this error, refer to the `patches/` folder and apply the fix to `solver.py` in your Conda environment before proceeding.

6. **Run AI Inference:**

   ```bash
   python mimickit/run.py --mode test --num_envs 1 --engine_config data/engines/newton_engine.yaml --env_config data/envs/amp_g1_vam_env.yaml --agent_config data/agents/amp_g1_vam_agent.yaml --visualize true --model_file checkpoint/model.pt
   ```

   The pre-trained AI model is included at `checkpoint/model.pt`; you do not need to train a model before trying the project.

7. **Virt-A-Mate Setup:**

   Load the following plugins in Virt-A-Mate:

   - `Receiver_mimickit_public.cs` — **Person plugin** (receives AI bone motion via UDP 9998).
   - `VRhand_mimickit_public.cs` — **Scene plugin** (sends VR controller and HMD tracking data via UDP 9999).
   - `Atom_sync_mimickit_public.cs` — **Scene plugin** (sends furniture Atom positions, rotations, and collider shapes via UDP 9001/9002).

   > ⚠️ **Important — Scene Alignment:** Before running the simulation, verify the following in VaM:
   > - The **Person Atom's Main Controller** (at the character's feet) has **world position `(0, 0, 0)`** and **rotation `(0, 0, 0)`**.
   > - The **ground Atom's Y position (height) is `0`**.
   >
   > Incorrect position or rotation alignment can cause furniture locations in Newton to differ from their positions relative to the character in VaM.

   **Network Configuration:**
   - **Single-PC Setup:** The default IP addresses (`127.0.0.1`) and UDP ports should work without any changes, provided there are no port conflicts.
   - **Dual-PC Setup:** Configure the IP addresses and UDP ports through the plugin UIs and Newton configuration file to match your local network environment.


8. **Done!** Once Newton and VaM are running and synchronized, the AI character will start moving. You can interact with it using VR controllers or test Follow Mode in Desktop Mode.

---

## Repository Structure & Modified Files

The following files are additions or modifications specific to this fork:

```text
MimicKit_VaM/
├── mimickit/
│   ├── envs/
│   │   ├── env_builder.py           # [Modified] Register the custom VaM environment
│   │   ├── amp_env_vam.py           # [Added] AMP environment, observations and resets
│   │   ├── vam_environment_boxes.py # [Added] Furniture transforms (9001) and shapes (9002)
│   │   ├── collider_simplifier.py   # [Added] Simplify and save received collider shapes
│   │   ├── vam_vr_tracking.py       # [Added] VR hand/HMD input (9999) and transforms
│   │   ├── vam_vr_body.py           # [Added] VR body/arm IK and dynamic capsules
│   │   ├── vam_vr_physics.py        # [Added] PD interactions and trigger-based dragging
│   │   ├── vam_vr_assets.py         # [Added] Generated XML assets for VR interaction
│   │   ├── vam_udp_sender.py        # [Added] Character motion output (9998)
│   │   └── task_location_env_vam.py # [Added] VaM-specific target location task
│   └── learning/
│       ├── agent_builder.py         # [Modified] Register inference agent
│       ├── base_agent.py            # [Modified] Support inference agent integration
│       ├── inference_agent.py       # [Added] Inference-only agent
│       └── inference_model.py       # [Added] Inference-only model
├── data/                             # Configuration, reference motions and assets
│   └── assets/
│       └── objects/
│           └── vam_env_boxes/        # Registered furniture collider assets
├── checkpoint/
│   └── model.pt                      # Pre-trained AI model
└── ...
```

---

## License & Attributions

- **Repository Code & Plugins:** Distributed under the **MIT License**.
- **Modified MimicKit Framework Files:** Files derived from the upstream MimicKit project are distributed under the **Apache License 2.0**, with authorship and modifications documented at the top of each file.

---

## Original MimicKit Documentation

For complete documentation on methods (DeepMimic, AMP, AWR, ASE, LCP, ADD, SMP) and the upstream codebase by Xue Bin Peng, see the official [xbpeng/MimicKit](https://github.com/xbpeng/MimicKit) repository.
