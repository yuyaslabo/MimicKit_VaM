# VaM_Physical_AI_from_Mimickit

Run physical AI trained with MimicKit inside Virt-A-Mate (VaM). Interact with the AI via VR controllers (push and grab).

> 💡 **Fork Notice:** This repository is a fork of the official [MimicKit](https://github.com/xbpeng/MimicKit) framework, combining core motion imitation with custom environment files and plugins for real-time UDP inference in Virt-A-Mate.

> ⚠️ **Disclaimer:** This is an unofficial community project. It is not affiliated with, associated with, or endorsed by MimicKit or MeshedVR.

📺 **DEMO:** [https://youtube.com](https://youtube.com) | [https://youtube.com](https://youtube.com)

---

## Overview

This project runs AI inference via **Newton** in the background while Virt-A-Mate runs. **A pre-trained physical AI model is included for immediate use.**

- **Desktop Mode:** Runs on a single PC. You can test the AI's **follow feature** even without a VR headset, although VR-based physical interactions are unavailable.
- **VR Mode:** Requires substantial computing power—typically a separate dedicated PC with an NVIDIA GPU for AI inference alongside the VaM PC. *(An RTX 4070 Ti single-PC setup proved insufficient during testing.)*
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

- IP addresses and ports can be configured for single-PC or dual-PC setups.
- **VaM side:** Configure VR hand outbound and character bone inbound settings from the plugin UI.
- **Newton/MimicKit side:** Adjust UDP settings in `data/envs/amp_g1_vam_env.yaml` (including `vam_udp_ip`, `vam_udp_port`, and `vrhand_udp_port`).
- **VaM plugins:**
  - `MimickitReceiver.cs` — **Person plugin** for receiving AI character motion.
  - `VRhand_Mimickit2.cs` — **Scene plugin** for VR controller/HMD data and controls.
  - `Atom_sync_mimickit_public.cs` — furniture Atom transform and collider shape synchronization.

### Furniture Synchronization

`Atom_sync_mimickit_public.cs` sends furniture Atom data from VaM to Newton:

- **UDP 9001:** Furniture Atom position and rotation updates.
- **UDP 9002:** Furniture Atom collider shape data, sent when the plugin starts and when new furniture Atoms are added to the scene.

### Registering Furniture in Newton (First Use)

Each furniture type must be registered the first time you use it. Registration is automatic, but **Newton must be restarted after the first registration** before that furniture is ready for normal use.

1. Start **Newton** and **Virt-A-Mate**.
2. Make sure the AI character is running and its motion is synchronized with the character in VaM.
3. Add the furniture Atom you want to use to the VaM scene.
4. Newton automatically processes the received collider shapes, simplifies them, and saves the furniture collider assets.
5. **Restart Newton** to complete the furniture setup.

Registered furniture can be reused in later sessions without repeating this process. The saved assets are located in:

```text
data/assets/objects/vam_env_boxes/
```

---

## VR Controls

Use a **double trigger press (×2)** on the corresponding VR controller. Perform these controls when not interacting with or grabbing an object.

| Controller | Action | Result |
| --- | --- | --- |
| **Left — double trigger** | Point at empty space and double press | Toggle AI target following **ON / OFF** (**Follow / Stay**). |
| **Right — double trigger** | Point in the desired direction and double press | Switch the AI's facing direction to the direction indicated by the controller. |
| **Right — double trigger** | Point at the ground and double press | Switch back to having the AI **look toward you**. |

---

## Setup Instructions

### Prerequisites

1. **Git:** Install from [git-scm.com](https://git-scm.com).
2. **Miniconda / Anaconda:** Install from [anaconda.com](https://anaconda.com).

### Installation & Execution

Run the following commands in **Anaconda Prompt**:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yuyaslabo/MimicKit_VaM
   cd MimicKit_VaM
   ```

2. **Create and activate the Python environment:**

   ```bash
   conda create --name mimickit python=3.11 -y
   conda activate mimickit
   ```

3. **Install Newton Simulator Backend (v1.0.0):**

   ```bash
   pip install newton==1.0.0
   ```

4. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   > **Troubleshooting (`J_ki` Error):** Depending on your GPU architecture (e.g., RTX 3090), a `J_ki`-related Warp compilation error may occur. If you encounter this error, refer to the `patches/` folder and apply the fix to `solver.py` in your Conda environment before proceeding.

5. **Run AI inference:**

   ```bash
   python mimickit/run.py --mode test --num_envs 1 --engine_config data/engines/newton_engine.yaml --env_config data/envs/amp_g1_vam_env.yaml --agent_config data/agents/amp_g1_vam_agent.yaml --visualize true --model_file checkpoint/model.pt
   ```

6. **Set up Virt-A-Mate:** Load `MimickitReceiver.cs` as a **Person plugin** and `VRhand_Mimickit2.cs` as a **Scene plugin**. Load `Atom_sync_mimickit_public.cs` to enable furniture synchronization.

---

## Repository Structure & Modified Files

The following files are additions or modifications specific to this fork:

```text
MimicKit_VaM/
├── mimickit/
│   ├── envs/
│       ├── env_builder.py           # [Modified] Register the custom VaM environment
│       ├── amp_env_vam.py           # [Added] AMP environment, observations and resets
│       ├── vam_environment_boxes.py # [Added] Furniture transforms (9001) and shapes (9002)
│       ├── collider_simplifier.py   # [Added] Simplify and save received collider shapes
│       ├── vam_vr_tracking.py       # [Added] VR hand/HMD input (9999) and transforms
│       ├── vam_vr_body.py           # [Added] VR body/arm IK and dynamic capsules
│       ├── vam_vr_physics.py        # [Added] PD interactions and trigger-based dragging
│       ├── vam_vr_assets.py         # [Added] Generated XML assets for VR interaction
│       ├── vam_udp_sender.py        # [Added] Character motion output (9998)
│       └── task_location_env_vam.py # [Added] VaM-specific target location task
│   └── learning/
│       ├── agent_builder.py         # [Modified] Register inference agent
│       ├── base_agent.py            # [Modified] Support inference agent integration
│       ├── inference_agent.py       # [Added] Inference-only agent
│       └── inference_model.py       # [Added] Inference-only model
├── data/                       # Config files, reference motions and assets
│   └── assets/objects/vam_env_boxes/ # Registered furniture collider assets
├── checkpoint/
│   └── model.pt                 # Pre-trained AI model
└── ...
```

---

## License & Attributions

- **Repository Code & Plugins:** Distributed under the **MIT License**.
- **Modified MimicKit Framework Files:** Files derived from the upstream MimicKit project are distributed under the **Apache License 2.0**, with authorship and modifications documented at the top of each file.

---

## Original MimicKit Documentation

For complete documentation on methods (DeepMimic, AMP, AWR, ASE, LCP, ADD, SMP) and the upstream codebase by Xue Bin Peng, see the official [xbpeng/MimicKit](https://github.com/xbpeng/MimicKit) repository.
