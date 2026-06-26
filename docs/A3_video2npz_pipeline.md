# A3 视频 → npz 转换链路

把人类打球视频转换成 A3 人形机器人可训练的运动 npz。链路横跨两个仓库：

- **上游** 视频 → SMPL-X → A3 `.pkl`
- **下游**  A3 `.pkl` → 训练用 `.npz`

```
[视频] --PromptHMR--> [SMPL-X] --GMR--> [A3 pkl] --pkl_to_npz--> [npz] --edit--> [no_yaw_float_z] --floor_align--> [floor_ik npz]
  原始视频              smplx.npz       robot_motion       clipX_a3.npz    去yaw/居中/抬z          逐脚IK贴地
```

最终产物：`data/A3/<NAME>_floor_ik.npz`，可直接喂给 whole_body_tracking 训练。

---

## 环境

| 阶段 | conda 环境 | 说明 |
|------|-----------|------|
| Step 1 (PromptHMR) | `phmr` | `scripts/extract_pose.py` 内部 subprocess 会自动切换；只需保证已安装 |
| Step 2 (GMR 重定向) | `a3_tt` | 该机器上没有 README 里写的 `gmr` 环境；用 `a3_tt`（含 `general_motion_retargeting` 依赖） |
| Step 3–5 (下游) | `a3_tt` | 含 `isaaclab` + `whole_body_tracking` + `mujoco` |

---

## Step 0 —— 一次性 GMR 配置修复（已于 2026-05-19 应用）

> 仅首次或排查脚踝朝向/浮脚问题时需要。已应用则跳过。

**问题**：原始 GMR 配置把 A3 的 `*_ankle_roll_link` 映射到 SMPL-X 的 `*_foot`（解剖学上是脚尖），导致系统性 ~11° 脚部 roll 偏差和明显的浮脚。

**修复文件**：`third_party/GMR/general_motion_retargeting/ik_configs/smplx_to_a3.json`

4 处编辑（两个 `ik_match_table` 块）：
- `"left_foot"` → `"left_ankle"`（line 77 / 304）
- `"right_foot"` → `"right_ankle"`（line 125 / 347）
- table1 `rot_w`: 10 → 15（line 79, 127）
- table2 `rot_w`: 50（不变，line 306, 349）

然后同步到嵌套副本（`video2robot/_init_gmr` 读的是嵌套副本，不是顶层那份，两者必须一致）：

```bash
cp /home/agiuser/p2/table_tennis_x2/third_party/GMR/general_motion_retargeting/ik_configs/smplx_to_a3.json \
   /home/agiuser/p2/table_tennis_x2/video2robot/third_party/GMR/general_motion_retargeting/ik_configs/smplx_to_a3.json
```

当前状态已验证：`left_ankle_roll_link → left_ankle`、`right_ankle_roll_link → right_ankle`，对称且正确。

---

## Step 1 —— 视频 → SMPL-X（PromptHMR）

前置：把待处理视频放到项目目录 `data/<NAME>/original.mp4`。

```bash
cd /home/agiuser/p2/table_tennis_x2/video2robot
python scripts/extract_pose.py --project data/<NAME>
# 静态相机可加 --static-camera 跳过 SLAM
```

**输出**：
- `data/<NAME>/smplx.npz` —— 第一个 track 的 SMPL-X 运动
- 多人时还会有 `smplx_track_N.npz` + `smplx_tracks.json`
- 中间产物：`results.pkl`、`world4d.glb/.mcs` 等（体积很大）

输入视频字段：`smplx.npz` 内含 `mocap_frame_rate`（通常 30），供下一步对齐用。

---

## Step 2 —— SMPL-X → A3 pkl（GMR 重定向）

```bash
conda activate a3_tt
cd /home/agiuser/p2/table_tennis_x2/video2robot
python scripts/convert_to_robot.py --project data/<NAME> --robot a3 --no-twist
```

要点：
- `--robot a3`（也支持 `a3_paddle`，带球拍的 MJCF）。`SUPPORTED_ROBOTS` 已注册这两个。
- `--no-twist`：TWIST 兼容输出只对 `unitree_g1` 生效，A3 无意义，加上避免误触。
- `--fps` 默认 0 = 保持原始帧率（推荐，这里通常是 30Hz）。
- 人身高被强制覆盖为 **1.75m**（`retargeter.py:124`），忽略不可靠的 betas 估计。

**输出**：`data/<NAME>/robot_motion_track_1.pkl`（track 1 会另存别名 `robot_motion.pkl`）。

pkl 结构：
```python
{
  "fps": 30.0,
  "robot_type": "a3",
  "num_frames": N,
  "human_height": 1.75,
  "root_pos":  (N, 3),
  "root_rot":  (N, 4),   # 四元数 xyzw
  "dof_pos":   (N, 29),  # 关节顺序 waist→arms→legs
  "local_body_pos": (N, num_bodies, 3),
  "link_body_list": [...],
}
```
`_build_robot_motion` 末尾会做一次整体抬高，使最低 body 的 z 落到 0。

---

## Step 3 —— A3 pkl → IsaacLab npz（30→50 Hz 上采样）

切到下游仓库。

```bash
conda activate a3_tt
cd /home/agiuser/p3/humanoid_table_tennis
# 该脚本若目标 npz 已存在会跳过 —— 重跑前先删
rm -f data/A3/<NAME>_a3.npz data/A3/<NAME>_a3_no_yaw_float_z.npz data/A3/<NAME>_a3_floor_ik.npz
python scripts/pkl_to_npz_a3.py \
    --input_file /home/agiuser/p2/table_tennis_x2/video2robot/data/<NAME>/robot_motion_track_1.pkl \
    --input_fps 30 --output_fps 50 \
    --output_name <NAME>_a3 \
    --output_file_dir_path /home/agiuser/p3/humanoid_table_tennis/data/A3 \
    --headless
```

**输出**：`data/A3/<NAME>_a3.npz`

注意：pkl 列顺序是 waist→arms→legs，npz 用 byd 顺序；脚本内部 `joint_names` 负责重排，已验证 `dof_pos` → `joint_pos` 逐关节匹配到 0.0000 rad（29 个关节全对）。

---

## Step 4 —— 去初始 yaw + xy 居中 + z 浮起

```bash
python tools/edit_motion_npz_a3.py --input_file data/A3/<NAME>_a3.npz
```

**输出**：`data/A3/<NAME>_a3_no_yaw_float_z.npz`（后缀 `_no_yaw_float_z` 硬编码，可用 `--output_file` 覆盖）。

把首帧朝向归零、水平居中、整体抬到地面之上，为下一步贴地做准备。

---

## Step 5 —— 逐脚 IK 贴地（floor align）

```bash
python tools/floor_align_a3.py \
    --input_file data/A3/<NAME>_a3_no_yaw_float_z.npz \
    --mode ik --ground_margin 0.0
```

**输出**：`data/A3/<NAME>_a3_floor_ik.npz` ← **最终训练用文件**

参数：
- `--ground_margin` 默认 0.005（5mm 安全余量）；用 `0.0` 得到最紧贴地（脚 mesh z 在 ±1mm 内）。
- `--mode` 默认 `ik`（另有 `lift`）。
- 调参旋钮：`--stance_height_thresh`(0.08)、`--stance_vel_thresh`(0.15)、`--stance_min_run`(5)、`--no_flatten_stance`（回退到旧的最低顶点法）。

**算法核心**（见 floor_align_a3.py）：
1. **支撑检测**：每帧每脚分为 planted（高度 < 0.08m 且 |竖直速度| < 0.15 m/s）vs swing。
2. **去抖**：`min_run=5`，去掉短于 5 帧的误判段，消除单帧“颠起放下”。
3. **连续混合权重**：从 0（swing，源关节不动）平滑过渡到 1（完全压平脚底），消除支撑/摆动边界的 50mm 跳变。
4. **摆动脚不动**：权重=0 时源腿关节原样保留，不对摆动脚做 IK（避免 68° 脚踝突跳）。
5. **支撑脚**：用 2 残差 DLS 在 6 个腿 DOF 上把脚尖+脚跟两个底角都压到 `ground_margin`。
6. **防穿模抬升**：逐帧整体刚性抬高，保证最低脚 ≥ 0。

> 关键陷阱：脚 mesh 底面在脚踝 body 原点下方约 6.9cm（geom 中心下约 5cm）。**永远不要把 geom 中心贴到 z=0，也不要对摆动脚做 IK 贴地。**

> 重要认知：A3 重定向出现的“踮脚/悬空”大多是**对源动作的忠实还原**（视频里人单脚站立、另一脚脚尖虚点），不是关节映射或 GMR 配置 bug。修复应在 `floor_align_a3.py` 的 stance-flatten，而不是改 GMR 配置（改配置会破坏左脚和其他 clip）。

---

## 可视化 / 验证

```bash
conda activate a3_tt
cd /home/agiuser/p3/humanoid_table_tennis
python tools/visual_motion_npz_a3.py --motion data/A3/<NAME>_a3_floor_ik.npz
```

可视化器只从 `joint_pos` + root pose 做 FK，忽略非 root 的 `body_pos_w`。


## 流程

```bash
# 1. 视频→SMPL-X      (video2robot, phmr)
python scripts/extract_pose.py --project data/<NAME>
# 2. SMPL-X→A3 pkl    (video2robot, a3_tt)
python scripts/convert_to_robot.py --project data/<NAME> --robot a3 --no-twist
# 3. pkl→npz          (humanoid_table_tennis, a3_tt)
rm -f data/A3/<NAME>_a3*.npz
python scripts/pkl_to_npz_a3.py --input_file .../robot_motion_track_1.pkl --input_fps 30 --output_fps 50 --output_name <NAME>_a3 --output_file_dir_path .../data/A3 --headless
# 4. 去yaw/浮z
python tools/edit_motion_npz_a3.py --input_file data/A3/<NAME>_a3.npz
# 5. 贴地（最终）
python tools/floor_align_a3.py --input_file data/A3/<NAME>_a3_no_yaw_float_z.npz --mode ik --ground_margin 0.0
```
