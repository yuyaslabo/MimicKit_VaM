# VaM_Physical_AI_from_Mimickit

Run physical AI trained with Mimickit inside Virt-A-Mate (VaM). Interact with the AI via VR controllers (push and grab).

> 💡 **Fork Notice:** This repository is a fork of the official [MimicKit](https://github.com/xbpeng/MimicKit) framework, combining core motion imitation with custom environment files and plugins for real-time UDP inference in Virt-A-Mate.

⚠️ **Disclaimer:** This is an unofficial community project. It is not affiliated with, associated with, or endorsed by Mimickit or MeshedVR.

📺 **DEMO:** [https://youtube.com] | [https://youtube.com]

---

## Overview

This project runs AI inference via **Newton** in the background while Virt-A-Mate runs. **A pre-trained physical AI model is included for immediate use.**

* **Desktop Mode:** Works on a single PC, though physical interactions inside Virt-A-Mate are unavailable.
* **VR Mode:** Requires heavy computing power—typically a separate dedicated PC with an NVIDIA GPU for AI inference alongside the VaM PC. *(An RTX 4070 Ti single-PC setup proved insufficient during testing).*
* **Hardware Compatibility:** Tested on **Meta Quest 3 OpenXR**. Other HMD controller tracking has not been verified.

---

## Network & Components

Data loops via UDP: `VaM VRhand HMD/environment` ➔ `127.0.0.1 UDP 9999/9001/9002` ➔ `Newton` ➔ `127.0.0.1 UDP 9998` ➔ `VaM Bone`

* IP addresses and ports are fully configurable for dual-PC or local setups.
* **Virt-A-Mate Side:** Configure VR hand outbound and bone inbound settings directly from the plugin UI.
* **Newton/Mimickit Side:** Adjust settings in `data/envs/amp_g1_vam_env.yaml` (`vam_udp_ip`, `vam_udp_port`, and `vrhand_udp_port`).
* **Included Plugins:** `MimickitReceiver.cs` (**Person plugin**) and `VRhand_Mimickit2.cs` (**Scene plugin**).

---

## Setup Instructions

### Prerequisites
1. **Git**: Install from [git-scm.com](https://git-scm.com).
2. **Miniconda / Anaconda**: Recommended lightweight package manager from [anaconda.com](https://anaconda.com).

### Installation & Execution
Run the following in your **Anaconda Prompt** :

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
   > **Troubleshooting (`J_ki` Error):** Depending on your GPU architecture (e.g., RTX 3090), a `J_ki`-related Warp compilation error may occur. If you encounter this error, please refer to the `patches/` folder and apply the fix to `solver.py` in your Conda environment before proceeding.

5. **Run AI Inference:**
   ```bash
   python mimickit/run.py --mode test --num_envs 1 --engine_config data/engines/newton_engine.yaml --env_config data/envs/amp_g1_vam_env.yaml --agent_config data/agents/amp_g1_vam_agent.yaml --visualize true --model_file checkpoint/model.pt
   ```

6. **Virt-A-Mate Setup:** Load `MimickitReceiver.cs` as a **Person plugin** and `VRhand_Mimickit2.cs` as a **Scene plugin** in Virt-A-Mate.


## Repository Structure & Modified Files

本フォークリポジトリにおける、独自システムの追加・改変ファイル構成です。

```text
MimicKit/
├── mimickit/
│   └── envs/
│       ├── env_builder.py          # [改変] 独自環境「amp_env_vam.py」の追加登録
│       ├── amp_env_vam.py          # [追加] AMP環境、観測、初期化・タスク/リセット処理
│       ├── vam_environment_boxes.py # [追加] UDP 9001によるオブジェクト位置同期 UDP9002によるシェイプ登録
│       ├── vam_vr_tracking.py      # [追加] UDP 9999によるVR手/HMD受信、座標変換・配置
│       ├── vam_vr_body.py          # [追加] プレーヤー胴体/腕IKとカプセルの動的配置
│       ├── vam_vr_physics.py       # [追加] PDコントローラによる外力制御、トリガーでのドラッグ処理
│       ├── vam_vr_assets.py        # [追加] 手/HMD/胴体のXMLアセットの自動生成
│       ├── vam_udp_sender.py       # [追加] UDP 9998によるVaMへのリアルタイムモーション送信
│       └── task_location_env_vam.py # [追加] 既存ファイルをVaM環境向けにパッケージ内配置
├── data/ #設定ファイルとオリジナル 参照モーション
├── checkpoint/
│   └── model.pt #学習済みモデル
```

## License & Attributions

* **Repository Code & Plugins:** Distributed under the **MIT License**.
* **Modified Mimickit Framework Files:** Files inside `mimickit/envs` derive from the upstream Mimickit project and are distributed under the **Apache License 2.0**, with authorship and modifications documented at the top of each file.

---

# Original Mimickit Documentation
For complete documentation on methods (DeepMimic, AMP, AWR, ASE, LCP, ADD, SMP) and the upstream codebase by Xue Bin Peng, please refer to the official [xbpeng/MimicKit](https://github.com/xbpeng/MimicKit) repository or check out the full details in the referenced web document.
