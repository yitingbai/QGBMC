# QGBMC 最小重跑数据契约

## 任务绑定

- 研究问题：在不使用真实标签调参的条件下，质量引导的梯度调制是否能提高完整多视图聚类在干净数据和受控噪声下的稳定性，并且其收益是否来自质量分数本身。
- 任务类型：完整多视图无监督聚类、受控噪声鲁棒性、质量分数隔离消融和规模测试。
- 目标变量：训练阶段不使用类别标签。标签只在全部训练结束后用于计算 ACC、NMI、ARI 和 PUR。
- 运行模式：`measured`。不产生模拟或示意性结果。

## 数据集路线

### Caltech101-20

- 模式：`reused-public`。
- 用途：主方法、简单基线、质量消融、噪声实验和 COMPLETER 官方实现对比。
- 来源：COMPLETER 的官方 CVPR 2021 代码仓库。
- 仓库：https://github.com/XLearning-SCU/2021-CVPR-Completer
- 仓库提交：`624dd9fd10431802214fab1d7b34c44f5a0a92df`。
- 本地文件：`_external_baselines/COMPLETER/data/Caltech101-20.mat`。
- SHA256：`AA51F62A9F37E177FB74ABF7EAB1BBE8C9A160E831835E20C3FF292793AC5658`。
- 结构：2,386 个对齐样本，6 个视图，维度为 48/40/254/1984/512/928，20 类。
- 访问条件：公开 GitHub 仓库，仓库含许可证文件。数据的上游许可仍需在最终 DOI 包中单独核对。
- 稳定标识质量：稳定 GitHub URL 和冻结提交，无数据 DOI。

### Scene-15

- 模式：`reused-public`。
- 用途：第二个中小规模数据集，用于检查结论是否只出现在 Caltech101-20。
- 来源、仓库与提交：同上，COMPLETER 官方仓库提交 `624dd9f...`。
- 本地文件：`_external_baselines/COMPLETER/data/Scene-15.mat`。
- SHA256：`EC33B5E1F53EEA04CCACC8B7BDFC6091ED490B04214EB41FD83C05A0A173221F`。
- 结构：4,485 个对齐样本。实际视图数量和维度由转换程序读取并写入运行清单，不从论文或 README 推测。
- 访问条件与稳定标识质量：同 Caltech101-20。

### ALOI_1K

- 模式：`agent-discovered`、`reused-public`。
- 用途：回应大规模数据要求，替代审稿人示例中的 YTF50。该选择不声称与 YTF50 等价，只验证十万级样本的可运行性与资源开销。
- 数据索引仓库：https://github.com/UeharaSetsuYu/Multi-view-Datasets
- 索引仓库提交：`03e5a826bd0659e1e5dab71864c81b589fe12f05`。
- 下载路由：索引仓库公开 Google Drive 文件夹中的 `ALOI_1K.mat`，文件 ID `1cMEk52vZnok-Hk0ez_yyTLNJgYgiUPyC`。
- 本地文件：`_external_datasets/ALOI_1K.mat`。
- SHA256：`C108BD9D6FDC924EE6BCB1A937EA0A8F6FFB75BBA047D75B2B9D219E7F8A4F57`。
- 结构：110,250 个对齐样本，4 个视图，维度为 125/77/13/64，1,000 类。
- 访问条件：公开下载链接；上游 ALOI 数据许可和特征再发布条件需要在 DOI 归档前核对。
- 稳定标识质量：稳定 GitHub 索引和 Google Drive 文件 ID，无数据 DOI。

## 本次运行使用的文件

- 用户指定的源代码目录：`C:/Users/zhang/Desktop/scientific report/sn-article-template/QGBMC/`。该目录含最小 QGBMC 实现、配置、测试和一个明确标注为 synthetic smoke test 的 `demo.npz`。本次运行使用其代码作为起点，但不把 `demo.npz` 作为论文结果。
- `data/caltech101_20.npz`：由 `Caltech101-20.mat` 的 `X` 和 `Y` 显式转换，SHA256 为 `84196640377BE343CDC32677B3A8FF1C94296749DEE17CB3F49706E3A47A4097`。
- `data/scene15.npz`：由 `Scene-15.mat` 的 `X` 和 `Y` 显式转换，SHA256 为 `C6C6320B29ECBD601D18BFAFA9AF2058A46DBE9B3D218A6A32B49EE8EF91EDD`。
- `data/caltech101_20_completer2.npz`：从上述 Caltech 文件确定性选取官方 COMPLETER 使用的第 4/5 个视图（1,984/512 维），用于相同输入的公平对照；SHA256 `3FC876850ED035604A55479BD6B0AE0F128A32870D3F8290719121F2C999A686`。
- `data/scene15_completer2.npz`：从上述 Scene 文件确定性选取官方 COMPLETER 使用的第 1/2 个视图（20/59 维），用于相同输入的公平对照；SHA256 `8F0572C2F9747FDFF77430B10FBF6E3F8F9DC583FBF4CEA8157ED8FB949C31F9`。
- `data/aloi_1k.npz`：由 `ALOI_1K.mat` 的 `fea` 和 `gt` 显式转换，SHA256 为 `A30C1B1B390E7E458C9444D2815109EEB27A9294C5EB8540B6DDA7A91B8328DF`。
- 类别标签只用于训练完成后的外部评价，不进入网络训练、早停、参数选择或模型挑选。

## 划分协议

- 多视图聚类采用完整数据集转导式训练，不创建使用标签的验证集。
- 所有视图保持样本一一对齐。转换脚本必须验证每个视图样本数相同、标签长度一致、数值有限。
- 参数不根据 ACC/NMI/ARI/PUR 选择。主配置在正式运行前固定，并跨数据集复用；敏感性实验单独报告，不用于挑选主表最优结果。
- COMPLETER 使用其官方数据与官方完整视图设置。若官方代码的训练协议必须修改以适配当前 PyTorch，只允许做兼容性修复，并保存补丁。

## 预处理契约

- 每个视图独立按特征维度进行 z-score 标准化，标准差小于 `1e-8` 的特征使用 1 作为分母。
- 所有标准化统计量和噪声掩码随运行保存。
- 干净实验不注入噪声。
- 噪声实验至少包含两类：选定视图的元素级高斯替换，以及样本级整行高斯替换。污染率表示被替换的元素或样本比例，取值、受污染视图和随机种子写入运行配置。
- 不把随机投影构造的伪多视图用于主实验，也不以单视图 MNIST/CIFAR-10 冒充标准多视图数据。

## 复用与发表说明

- 本运行只有在代码实际处理上述文件、保存原始日志、配置、数据哈希和结果后，才可标记为 measured。
- 本次最小重跑用于验证审稿人关切，不预设必须复现原稿中的具体数值。
- ALOI_1K 是大规模替代数据集，回复信必须明确这是针对规模问题新增的证据，不能写成审稿人指定的 YTF50 已完成。
- 未解决风险：数据集缺少 DOI；ALOI 上游许可需要核验；COMPLETER 是不完整多视图方法，完整视图比较的适用性需在报告中说明。

