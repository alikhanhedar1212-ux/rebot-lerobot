# reBot LeRobot：重力补偿、主从跟随与 ACT 抓取

本项目用于两套 Seeed reBot Arm B601-DM 的主臂重力补偿、双适配器主从跟随，以及基于双 RealSense 相机的 ACT 数据采集、训练和真机单次安全抓取。第一组设备默认映射为主臂 `device_index=1`、从臂 `device_index=0`。

## 安全要求

1. 启动前清空机械臂运动范围，确认实体急停可用。
2. 首次接线或更换电脑后，先做相机检查和 CAN 只读检查。
3. 跟随和抓取前，让机械臂处于安全、且与训练数据接近的起始姿态。
4. 不要同时运行多个占用 DM-USB2FDCAN 或 RealSense 的程序。
5. ACT 首次真机测试必须有人守在急停旁，先在空旷环境单次运行。

## 环境与硬件确认

所有命令从项目根目录执行：

```bash
cd /path/to/rebot_lerobot
source deploy/runtime_env.sh
```

`runtime_env.sh` 会根据脚本位置确定项目目录，不依赖用户名或固定安装路径。机器相关配置集中在：

- `config/machine.env`：主从臂 USB-CAN 索引、相机序列号和本机视频节点。
- `config/cameras.json`：采集和 ACT 推理使用的相机参数。

关闭相机占用程序并检查相机、CAN：

```bash
pkill -x realsense-viewer 2>/dev/null || true
python view_cameras.py
python dual_arm/test_two_adapters_read.py
```

CAN 检查应显示主臂和从臂均为 `7/7 feedback`。若角色相反，只交换 `config/machine.env` 中的 `REBOT_LEADER_INDEX` 和 `REBOT_FOLLOWER_INDEX`。

## 主臂重力补偿

重力补偿让主臂进入 MIT 控制并持续发送补偿力矩，使主臂可以被人手拖动。它只用于主臂，不等同于从臂位置控制。

命令行启动：

```bash
python dual_arm/test_leader_gravity.py --device-index "$REBOT_LEADER_INDEX"
```

运行时缓慢扶住并移动主臂，观察关节是否平稳、方向是否正常。按 `Ctrl+C` 后程序停止补偿并执行退出清理。

网页启动：

```bash
./control_panel/start_control_panel.sh
```

浏览器打开终端显示的 `Local` 或 `Network` 地址，先点击“重新检测硬件”，确认主臂 `7/7` 后再启动重力补偿。结束时使用网页停止按钮；危险情况下直接按实体急停。

## 主从跟随

当前只使用两个独立 DM-USB2FDCAN 适配器：主臂一个、从臂一个。不再支持单适配器共享 14 个电机句柄的旧方案。

命令行启动：

```bash
./dual_arm/start_two_adapters_follow.sh
```

主臂启动重力补偿并读取 7 个关节位置，从臂使能后接收映射目标，以固定循环执行跟随。按 `Ctrl+C` 会停止补偿、失能从臂并关闭适配器。

网页操作顺序：

1. 启动 `./control_panel/start_control_panel.sh`。
2. 点击“重新检测硬件”，确认主臂和从臂均为 `7/7`。
3. 确认没有单独运行的重力补偿或从臂使能任务。
4. 启动主从跟随，先用小幅、低速动作验证方向。
5. 结束时点击停止主从跟随。

## ACT 完整流程

完整闭环是：确定任务和拍摄条件 → 遥操作采集示范 → 检查并清洗数据 → 训练 ACT → 选择 checkpoint → 准备离线权重 → 单次真机评估 → 根据失败情况补数据或调整训练。

### 1. 固定任务、相机和起始条件

当前任务文本为：`Pick up the green object and place it on the white paper.`。默认配置为：

- D405 腕部相机：`wrist`，640×480、30 FPS、YUYV。
- D435i 场景相机：`scene`，640×480、30 FPS、YUYV。
- 机械臂状态和动作采样频率：30 Hz。
- 每条示范 15 秒，示范之间复位 30 秒。

正式采集前应固定相机安装位置、桌面范围、物体与白纸的大致分布、光照和机械臂初始姿态。训练后若明显改变相机视角或任务条件，模型效果通常会下降。

### 2. 先采集一条测试数据

测试目录必须是新目录：

```bash
./dual_arm/record_pair1_act.sh \
  --dataset.root="$PROJECT_DIR/data/act_deploy_test" \
  --dataset.repo_id=local/act_deploy_test \
  --dataset.num_episodes=1
```

采集脚本会执行以下工作：

1. 加载统一运行环境和本机配置。
2. 检查 `realsense-viewer` 是否占用相机。
3. 设置两台相机的曝光、亮度和锐度参数。
4. 使用主臂索引作为 teleoperator、从臂索引作为 robot。
5. 主臂自动进入重力补偿，由操作者拖动主臂完成示范。
6. 同步记录 7 维关节状态、7 维动作和两路 RGB 视频。

示范动作应平稳完整：从相近起始姿态出发，接近物体，夹紧，移动到白纸，释放，并留出短暂停顿。不要保留镜头被遮挡、发生碰撞、动作做到一半结束或复位动作混入有效片段的数据。

### 3. 正式采集示范

默认正式数据集为 `data/act_green_to_paper`，共采集 15 个 episode：

```bash
./dual_arm/record_pair1_act.sh
```

脚本默认不覆盖已有目录。确认要在同一数据集上继续采集时，才使用：

```bash
./dual_arm/record_pair1_act.sh --resume=true
```

不要为了凑数量保留失败示范。示范需要保持任务逻辑和动作风格一致，同时让物体位置、抓取角度和运动轨迹有合理变化，避免模型只记住一条固定路径。

### 4. 检查和清洗数据

逐条查看视频、动作完整性和相机角色：

```bash
lerobot-dataset-viz \
  --repo-id local/act_green_to_paper \
  --root "$PROJECT_DIR/data/act_green_to_paper" \
  --episode-index 0 \
  --mode local \
  --display-compressed-images true
```

修改 `--episode-index` 逐条检查。典型坏数据包括：未抓到物体、物体掉落、动作中断、画面卡死或错位、相机被遮挡、从异常姿态开始。

删除坏 episode 时生成新数据集，不要直接删除 MP4、Parquet 文件或数据行。例如删除第 1 和第 3 条：

```bash
lerobot-edit-dataset \
  --repo_id local/act_green_to_paper \
  --root "$PROJECT_DIR/data/act_green_to_paper" \
  --new_repo_id local/act_green_to_paper_clean \
  --operation.type delete_episodes \
  --operation.episode_indices '[1, 3]'
```

清洗后的训练目录为 `data/act_green_to_paper_clean`。保留原始数据，便于重新选择 episode。

### 5. 训练 ACT

以下命令使用清洗后的数据训练 50,000 steps，batch size 为 2，每 10,000 steps 保存 checkpoint：

```bash
lerobot-train \
  --dataset.repo_id=local/act_green_to_paper_clean \
  --dataset.root="$PROJECT_DIR/data/act_green_to_paper_clean" \
  --policy.type=act \
  --policy.device=cuda \
  --output_dir="$PROJECT_DIR/outputs/act_green_to_paper_clean_bs2_50k" \
  --job_name=act_green_to_paper_clean_bs2_50k \
  --batch_size=2 \
  --num_workers=2 \
  --steps=50000 \
  --save_checkpoint=true \
  --save_freq=10000 \
  --log_freq=100 \
  --policy.push_to_hub=false
```

训练前检查 GPU：

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

若显存不足，先把 `--batch_size=2` 改为 `--batch_size=1`；若视频读取成为瓶颈，可根据 CPU 和内存调整 `--num_workers`。训练输出写入指定的 `outputs/...`，不会自动替换网页正在使用的模型。

训练期间观察 loss 是否下降，以及是否出现 CUDA OOM、数据读取错误或 NaN。最终不要只按训练步数选模型，应分别用保存的 checkpoint 在相同条件下做单次真机评估。

### 6. 选择并部署 checkpoint

当前安全抓取脚本使用：

```text
outputs/act_green_to_paper_bs2_50k/checkpoints/050000/pretrained_model
```

新模型训练完成后，确认目标 checkpoint 内存在完整的 `pretrained_model`，再修改 `dual_arm/run_act_single_grasp_safe.sh` 顶部的 `POLICY_DIR`。命令行和网页共用这个抓取入口，因此只需修改这一处。

ACT 创建 ResNet-18 视觉骨干时还需要本地预训练权重：

```text
~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth
```

离线运行前校验：

```bash
sha256sum ~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth
```

正确 SHA256：

```text
f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec
```

缺少该文件时，程序会尝试联网下载；无网络或 SSL 异常会让抓取进程在模型初始化阶段退出。

### 7. 单次安全抓取

每次测试使用新的 run ID：

```bash
./dual_arm/run_act_single_grasp_safe.sh 001_safe
```

该入口会：

1. 加载指定 ACT checkpoint 和两路相机。
2. 只连接从臂，不连接主臂遥操作器。
3. 创建一条 20 秒评估记录，保存到 `data/eval_act_green_to_paper_<run-id>`。
4. 启用单次抓取安全逻辑，包括 5 秒静止判断、动作变化阈值、起步幅度阈值和夹爪开合阈值。
5. 结束或异常退出时执行机械臂断开清理。

同名评估目录不会被覆盖，因此第二次运行必须换 run ID。首次测试时将物体和机械臂放到与训练分布一致的位置，并随时准备按实体急停。

网页抓取使用同一脚本。操作顺序：

1. 启动 `./control_panel/start_control_panel.sh`。
2. 重新检测硬件，确认从臂和两路相机可用。
3. 结束重力补偿、从臂单独使能和主从跟随。
4. 点击启动抓取。
5. 实时查看固定日志：

```bash
tail -F /tmp/rebot_act_green_to_paper_web_safe.log
```

网页按钮没有动作且终端无输出时，优先查看该日志。常见原因是 ResNet-18 权重缺失、相机节点错误或被占用、模型目录不存在、评估数据目录重名，以及 USB-CAN 角色配置错误。

### 8. 评估后迭代

每次记录物体初始位置、是否成功抓取、是否成功放置、失败阶段和所用 checkpoint。若失败集中在某类位置或角度，应补采对应条件下的高质量示范，再重新清洗和训练；若动作方向或相机画面错误，应先修正硬件配置，不能靠增加训练步数解决。

## 项目迁移简述

迁移到新电脑时复制项目目录和已打包的 Python 环境；新电脑只需准备兼容的 NVIDIA Driver、USB/udev 与 RealSense 系统支持，解压环境后更新 `config/machine.env` 和 `config/cameras.json`。其他 Conda 环境不会影响本项目，因为运行脚本固定使用项目内 `.conda/envs/lerobot-follow`。迁移后先完成 GPU、相机和 CAN 只读检查，再运行机械臂。
