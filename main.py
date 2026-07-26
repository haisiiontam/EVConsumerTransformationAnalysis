\documentclass[UTF8,a4paper,12pt]{ctexart}

\usepackage{geometry}
\usepackage{setspace}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{array}
\usepackage{float}
\usepackage{amsmath}
\usepackage{hyperref}
\usepackage[backend=biber,style=gb7714-2015]{biblatex}

\geometry{left=2.5cm,right=2.5cm,top=2.5cm,bottom=2.5cm}
\onehalfspacing
\setlength{\parindent}{2em}
\setlength{\parskip}{0pt}

\hypersetup{
  colorlinks=true,
  linkcolor=black,
  citecolor=black,
  urlcolor=black
}

\addbibresource{references.bib}
\graphicspath{{figures/}}

\title{\heiti 从燃油到电动：电动汽车消费转型的分类预测模型研究}
\author{队伍编号：DATA2600907}
\date{}

\begin{document}

\maketitle

\begin{abstract}
本文围绕 2026 年第五届“创新杯”大学生大数据挑战赛 B 题展开研究，基于 50000 条消费者记录，分别建立电动汽车采用意愿分类模型、家庭充电设施安装潜力预测模型和里程焦虑敏感性识别模型。首先，本文对数据进行缺失值填补、异常值识别、类别变量编码和数值变量标准化处理，并统一构建监督分类建模框架。其次，针对三个问题分别确定目标变量、特征集合与评价指标：问题一采用 CatBoost 三分类模型预测消费者采用意愿等级；问题二采用 CatBoost 二分类模型预测用户是否具备家庭充电条件，并通过阈值搜索优化分类性能；问题三将里程焦虑得分转化为二分类标签，采用 CatBoost 二分类 OOF 阈值搜索模型识别高里程焦虑用户。
\end{abstract}

\noindent\textbf{关键词：电动汽车；消费转型；分类预测；CatBoost；类别权重；阈值搜索}

\newpage
\tableofcontents
\newpage

\section{问题重述}

\subsection{问题背景}

电动汽车是交通能源转型的重要方向。消费者是否愿意从传统燃油汽车转向电动汽车，不仅受到收入、通勤距离、燃油支出等经济因素影响，也与充电基础设施可达性、环保意识、技术接受度、电动车知识以及里程焦虑等心理和认知因素密切相关。题目提供了一份包含 50000 条消费者记录的数据集，字段覆盖人口统计特征、通勤行为、当前车辆情况、充电条件、环境意识、技术接受度和电动汽车相关认知等多个维度，为建立数据驱动的消费转型预测模型提供了基础。

\subsection{问题提出}

根据题目要求，本文需要解决以下三个分类建模问题：

\begin{enumerate}
  \item 电动汽车采用意愿分类：根据消费者基本信息、出行行为、能源支出、充电便利性、环保意识和技术接受度等变量，预测消费者的电动汽车采用意愿等级，即 \texttt{Low}、\texttt{Medium} 或 \texttt{High}。
  \item 家庭充电设施安装潜力预测：根据城市类型、最近充电站距离、学历、收入、车辆年龄等变量，预测用户是否具备家庭充电条件。
  \item 里程焦虑敏感性识别：根据通勤距离、周出行距离、充电站可达性、电动车知识和电池更换担忧等变量，识别用户是否属于高里程焦虑群体。
\end{enumerate}

\section{符号与模型说明}

\subsection{符号说明}

为便于后续建模，本文对主要符号进行统一说明，如表 \ref{tab:symbols} 所示。

\begin{table}[H]
  \centering
  \caption{主要符号说明}
  \label{tab:symbols}
  \begin{tabular}{p{0.18\textwidth}p{0.72\textwidth}}
    \toprule
    符号 & 含义 \\
    \midrule
    $n$ & 样本数量，本文中 $n=50000$ \\
    $i$ & 用户样本编号，$i=1,2,\ldots,n$ \\
    $x_i$ & 第 $i$ 个用户的原始特征向量 \\
    $\tilde{x}_i$ & 经缺失值处理、编码和标准化后的特征向量 \\
    $x_{ij}$ & 第 $i$ 个样本在第 $j$ 个数值变量上的取值 \\
    $z_{ij}$ & 第 $j$ 个数值变量标准化后的取值 \\
    $Y_1$ & 问题一目标变量，表示电动汽车采用意愿等级 \\
    $Y_2$ & 问题二目标变量，表示是否具备家庭充电条件 \\
    $Y_3$ & 问题三目标变量，表示是否为高里程焦虑用户 \\
    $\hat{Y}_q$ & 第 $q$ 个问题的模型预测标签，$q=1,2,3$ \\
    $K_q$ & 第 $q$ 个问题的类别数 \\
    $p(k\mid x_i)$ & 样本 $x_i$ 属于类别 $k$ 的预测概率 \\
    $ACC_q$ & 第 $q$ 个问题的分类准确率 \\
    $Score$ & 三个问题的加权综合成绩 \\
    \bottomrule
  \end{tabular}
\end{table}

\subsection{统一分类建模框架}

三个问题均属于监督分类问题。设第 $q$ 个问题的数据集为
\begin{equation}
  D_q=\{(\tilde{x}_i,y_{qi})\}_{i=1}^{n},
\end{equation}
其中 $\tilde{x}_i$ 为预处理后的特征向量，$y_{qi}$ 为第 $q$ 个任务对应的真实标签。模型训练的目标是学习映射函数
\begin{equation}
  f_q:\tilde{x}_i\rightarrow \hat{y}_{qi},
\end{equation}
使预测标签 $\hat{y}_{qi}$ 尽可能接近真实标签 $y_{qi}$。题目以准确率为评分依据，因此本文采用准确率作为主要评价指标：
\begin{equation}
  ACC_q=\frac{1}{n_q}\sum_{i=1}^{n_q}I(\hat{y}_{qi}=y_{qi}),
\end{equation}
其中 $I(\cdot)$ 为示性函数，$n_q$ 为验证集中第 $q$ 个问题的样本数。

三个问题的最终加权得分为
\begin{equation}
  Score=100\times(0.2ACC_1+0.4ACC_2+0.4ACC_3).
\end{equation}
由于问题二和问题三各占 40\% 权重，后续模型优化时应重点关注 $ACC_2$ 与 $ACC_3$。

\section{数据预处理}

\subsection{数据来源与字段概况}

本文使用题目提供的 \texttt{B题数据集.csv} 作为建模数据。数据集共包含 50000 条样本和 23 个字段，其中既有年龄、收入、通勤距离、燃油支出、最近充电站距离等数值变量，也有学历、城市类型、当前车辆类型等类别变量，还包括采用意愿、家庭充电条件和里程焦虑得分等目标相关变量。

数据中的三个目标字段分别为：
\begin{enumerate}
  \item \texttt{ev\_adoption\_likelihood}：问题一目标变量，取值为 \texttt{Low}、\texttt{Medium}、\texttt{High}；
  \item \texttt{home\_charging\_available}：问题二目标变量，取值为 0 或 1；
  \item \texttt{range\_anxiety\_score}：问题三原始目标变量，取值范围为 1 到 10。
\end{enumerate}

\subsection{缺失值与异常值处理}

数据检查发现，学历、充电站可达性和电动车知识得分三个字段分别存在 500 个缺失值。考虑到缺失比例约为 1\%，本文采用如下处理方法：

\begin{enumerate}
  \item 类别型变量 \texttt{education\_level} 和 \texttt{city\_type} 等，缺失值单独设为 \texttt{Unknown} 类别，避免错误归入众数；
  \item 数值型变量 \texttt{charging\_station\_accessibility} 和 \texttt{ev\_knowledge\_score} 使用中位数填补，同时增加缺失指示变量，以捕捉缺失本身可能蕴含的"不愿回答"或"不清楚"等信息；
  \item 对连续变量进行范围检查。对于明显不合理的数值，采用分位数截尾或中位数替代，以降低异常值对模型的影响；
  \item 所有缺失填补参数均在每个训练折内单独计算，避免验证集信息泄露。
\end{enumerate}

\subsection{变量编码与标准化}

对学历、城市类型、当前车辆类型等无序类别变量，本文采用独热编码：
\begin{equation}
  c\rightarrow (d_1,d_2,\ldots,d_m),
\end{equation}
其中 $m$ 为该类别变量的取值个数，$d_j\in\{0,1\}$。

对收入、出行距离、燃油支出、充电距离和各类评分等连续变量，本文采用标准化处理：
\begin{equation}
  z_{ij}=\frac{x_{ij}-\mu_j}{\sigma_j},
\end{equation}
其中 $\mu_j$ 和 $\sigma_j$ 分别表示第 $j$ 个变量在训练集中的均值和标准差。标准化只在训练集上计算参数，并将同一参数应用于验证集，以避免数据泄露。

\subsection{交叉验证与划分策略}

为充分评估模型的泛化能力，本文采用分层 5 折交叉验证。将样本按目标变量分布分层划分为 5 个互斥折，每次取其中 4 折作为训练集、1 折作为验证集，循环 5 次。所有缺失填补参数、标准化参数、类别编码映射均在每个训练折内独立计算并应用于对应验证折，避免数据泄露。每折生成的验证集预测汇总为 OOF 预测，模型融合权重和分类阈值均在 OOF 预测上搜索确定，而非单次验证集结果。

\section{问题一：电动汽车采用意愿分类模型}

\subsection{问题一求解思路}

问题一要求根据消费者画像预测其电动汽车采用意愿，目标变量包含 \texttt{Low}、\texttt{Medium}、\texttt{High} 三个等级，属于三分类任务。本文采用赛题官方建议的 6 个特征作为输入：年收入（\texttt{annual\_income}）、学历（\texttt{education\_level}）、环保意识得分（\texttt{environmental\_awareness\_score}）、技术接受度得分（\texttt{technology\_affinity\_score}）、里程焦虑得分（\texttt{range\_anxiety\_score}）和月度燃油费用（\texttt{fuel\_expense\_per\_month}）。

模型选用 CatBoost 梯度提升决策树。CatBoost 通过 Ordered Boosting 机制有效缓解预测偏移问题，并原生支持类别特征，无需手动独热编码。对于三分类任务，设置损失函数为 \texttt{MultiClass}，直接输出三个类别的预测概率。

在验证方式上，将全部 50000 条数据按 80\%/20\% 比例进行分层随机划分，确保训练集与验证集中 \texttt{Low}、\texttt{Medium}、\texttt{High} 的分布比例与全量数据一致。模型在训练集上完成 300 轮迭代后，在验证集上评估分类准确率。

从影响机制看，电动汽车采用意愿主要由四类因素共同决定：第一类是经济压力因素，包括年收入和燃油支出；第二类是出行需求因素，包括日均通勤距离和当前车辆年龄；第三类是充电基础设施因素，包括充电站可达性和最近充电站距离；第四类是认知心理因素，包括环保意识、技术接受度、里程焦虑和电动车知识。



\subsection{问题一模型建立}

设问题一的类别集合为
\begin{equation}
  \mathcal{C}_1=\{\mathrm{Low},\mathrm{Medium},\mathrm{High}\}.
\end{equation}
为便于建模，将其编码为
\begin{equation}
  y_{1i}\in\{0,1,2\},
\end{equation}
其中 0、1、2 分别对应 \texttt{Low}、\texttt{Medium}、\texttt{High}。设预处理和特征工程后的输入向量为 $\tilde{x}_i$。

CatBoost 属于梯度提升决策树模型。设第 $t$ 轮迭代得到的模型为 $F_t(x)$，则模型通过不断加入新的弱学习器 $h_t(x)$ 修正前一轮预测误差：
\begin{equation}
  F_t(x)=F_{t-1}(x)+\eta h_t(x),
\end{equation}
其中 $\eta$ 为学习率。CatBoost 采用对称决策树作为基学习器，并使用 Ordered Boosting 机制生成每棵树的梯度估计，从而减少传统梯度提升中的预测偏移。

对于三分类任务，CatBoost 为每个样本输出三个类别的原始得分 $F_k(\tilde{x}_i)$，并通过 Softmax 函数转化为类别概率：
\begin{equation}
  P(Y_1=k\mid \tilde{x}_i)=
  \frac{\exp(F_k(\tilde{x}_i))}
  {\sum_{l\in\mathcal{C}_1}\exp(F_l(\tilde{x}_i))},
  \quad k\in \mathcal{C}_1.
\end{equation}
最终预测类别为概率最大的类别：
\begin{equation}
  \hat{Y}_1=\arg\max_{k\in\mathcal{C}_1}P(Y_1=k\mid \tilde{x}_i).
\end{equation}

模型训练采用多分类交叉熵损失函数：
\begin{equation}
  L_1=-\frac{1}{n}\sum_{i=1}^{n}\sum_{k\in\mathcal{C}_1}
  I(y_{1i}=k)\log P(Y_1=k\mid \tilde{x}_i).
\end{equation}

模型超参数设置如下：迭代数 300，树深度 6，学习率 0.05，随机种子 42。学历特征 \texttt{education\_level} 作为类别特征由 CatBoost 自动处理。

\subsection{问题一结果分析}

在 80\%/20\% 分层划分下，CatBoost 三分类模型在验证集上取得的准确率为 0.8592。验证集分类报告如表 \ref{tab:task1-report} 所示。

\begin{table}[H]
  \centering
  \caption{问题一验证集分类报告}
  \label{tab:task1-report}
  \begin{tabular}{lcccc}
    \toprule
    类别 & 精确率 & 召回率 & F1得分 & 样本数 \\
    \midrule
    Low    & 0.78 & 0.73 & 0.75 & 1650 \\
    Medium & 0.65 & 0.71 & 0.68 & 2416 \\
    High   & 0.90 & 0.90 & 0.90 & 5934 \\
    \midrule
    准确率 & \multicolumn{4}{c}{0.8592} \\
    \bottomrule
  \end{tabular}
\end{table}

从结果可以看出，模型对 \texttt{High} 类的识别能力最强（F1=0.90），这与 \texttt{High} 类样本占比较高（约 59\%）有关；对 \texttt{Medium} 类的识别能力最弱（F1=0.68），说明中间类别与两端的决策边界区分度不足。特征重要性分析显示，技术接受度（\texttt{technology\_affinity\_score}）和环保意识得分（\texttt{environmental\_awareness\_score}）对预测贡献最大，表明认知心理因素对消费者采用意愿有较强解释力。


\section{问题二：家庭充电设施安装潜力预测模型}

\subsection{问题二求解思路}

问题二要求预测消费者是否具备家庭充电条件，目标变量 \texttt{home\_charging\_available} 取值为 0 或 1，属于二分类问题。

家庭充电条件通常与居住环境、收入水平、城市类型、车辆使用状况和公共充电设施距离有关。需要注意的是，该变量更接近"当前是否具备家用充电条件"，而非"未来安装意愿"。因此，模型输出应理解为用户成为家用充电桩潜在客户的相对可能性。

家庭充电条件的影响因素之间存在复杂非线性关系。此外，问题二在评分中占 40\% 权重，是提升总成绩的重点。综合考虑变量混合类型、非线性交互和优化空间，本文选取 CatBoost 作为问题二的主模型，并在 OOF 预测上搜索最优分类阈值，以缓解样本不均衡带来的偏向性预测。

\subsection{问题二模型建立}

\subsubsection{CatBoost 二分类模型}

问题二采用 CatBoost 梯度提升决策树模型。CatBoost 通过有序提升克服预测偏移问题，并原生支持高基数类别变量。对样本 $\tilde{x}_i$，模型输出正类对数几率：
\begin{equation}
  \mathrm{logit}_2(\tilde{x}_i)=F(\tilde{x}_i),
\end{equation}
其中 $F(\tilde{x}_i)$ 为梯度提升累积输出。正类概率由 Sigmoid 函数转换：
\begin{equation}
  P(Y_2=1\mid \tilde{x}_i)=
  \frac{1}{1+\exp(-\mathrm{logit}_2(\tilde{x}_i))}.
\end{equation}

模型采用二分类交叉熵损失函数：
\begin{equation}
  L_2=-\frac{1}{n}\sum_{i=1}^{n}
  \bigl[y_{2i}\log P(Y_2=1\mid \tilde{x}_i)
  +(1-y_{2i})\log(1-P(Y_2=1\mid \tilde{x}_i))\bigr].
\end{equation}

由于正类样本占比较高，直接使用 0.5 作为分类阈值易导致模型偏向预测正类。因此，本文在 OOF 预测上搜索最优阈值：
\begin{equation}
  t_2^{*}=\arg\max_{t} ACC_{\mathrm{OOF}}(t),\quad t\in[0.30,0.70].
\end{equation}
最终预测为
\begin{equation}
  \hat{Y}_2=
  \begin{cases}
    1, & P(Y_2=1\mid \tilde{x}_i)\geq t_2^{*},\\
    0, & P(Y_2=1\mid \tilde{x}_i)<t_2^{*}.
  \end{cases}
\end{equation}

\subsubsection{特征构造}

问题二建模时删除 \texttt{home\_charging\_available} 防止目标泄露。为增强模型对安装潜力的刻画，构造以下组合特征：

\begin{enumerate}
  \item 充电缺口指数：$\displaystyle \frac{\texttt{nearest\_charging\_station\_km}}{\texttt{charging\_station\_accessibility}+1}$，充电站越远且越不便利时数值越高；
  \item 充电便利指数：$\displaystyle \frac{\texttt{charging\_station\_accessibility}}{\texttt{nearest\_charging\_station\_km}+1}$，反映基础设施综合便利程度；
  \item 燃油经济压力：$\displaystyle \frac{12\times\texttt{fuel\_expense\_per\_month}}{\texttt{annual\_income}+1}$，年燃油开销占收入比例；
  \item 换车紧迫度：$\texttt{vehicle\_age\_years}\times\texttt{fuel\_expense\_per\_month}$，高车龄高油费带来的换车动机。
\end{enumerate}

\subsection{问题二结果分析}

CatBoost 模型在问题二上的平均准确率约为 0.6498。验证集混淆矩阵如表 \ref{tab:confusion-p2} 所示。

\begin{table}[H]
  \centering
  \caption{问题二验证集混淆矩阵}
  \label{tab:confusion-p2}
  \begin{tabular}{lcc}
    \toprule
    真实类别 & 预测 0 & 预测 1 \\
    \midrule
    0 & 28 & 3474 \\
    1 & 65 & 6433 \\
    \bottomrule
  \end{tabular}
\end{table}

结果显示，模型仍倾向于预测正类，这与数据中正类样本占比较高有关。特征重要性分析显示，年收入、燃油支出、周出行距离、月能耗、月充电成本、车辆年龄和充电站距离是贡献最大的变量。其中，充电缺口指数和燃油经济压力等构造特征对预测有稳定正向贡献。

\subsection{本章小结}

本章针对家庭充电设施安装潜力建立了 CatBoost 二分类模型，引入领域驱动组合特征和分类阈值搜索。模型对正类样本识别较好，但对负类区分能力有限，是三个任务中优化空间最大的部分。

\section{问题三：里程焦虑敏感性识别模型}

\subsection{问题三求解思路}

问题三要求识别对电动汽车续航里程较为敏感的用户。原始里程焦虑得分为 1 到 10 的评分变量。根据题目建议，将其转化为二分类标签：
\begin{equation}
  Y_3=
  \begin{cases}
    1, & \texttt{range\_anxiety\_score}>5,\\
    0, & \texttt{range\_anxiety\_score}\leq 5.
  \end{cases}
\end{equation}
其中 $Y_3=1$ 表示高里程焦虑用户，$Y_3=0$ 表示低里程焦虑用户。

本文采用特征扩展与 OOF 交叉验证的建模策略。首先构建三种不同特征配置进行对比：配置 A 使用官方建议的 5 个特征（日均通勤距离、周出行距离、充电站可达性、电动车知识得分、电池更换担忧得分）；配置 B 在 A 的基础上，进一步纳入年收入、年龄、学历、城市类型、环保意识、技术接受度等 18 个安全扩展特征；配置 C 在 B 的基础上，添加手工构造的机制特征（如通勤比例、知识缺陷得分、环保—技术均值等 6 个特征），共 24 个特征。

为充分利用数据且避免验证集信息泄露，模型采用分层 5 折交叉验证生成 OOF（Out-of-Fold）预测。针对二分类任务，在 OOF 概率上搜索最优分类阈值（搜索范围 0.30--0.70，步长 0.005），以最大化 OOF 准确率。最终在开发集（80\% 数据）上使用最优特征配置和最优阈值重训模型，再在预留的验证集（20\% 数据）上做一次独立评估。

\subsection{问题三模型建立}

设样本 $i$ 的输入特征向量为 $\tilde{x}_i$，CatBoost 二分类模型输出正类对数几率：
\begin{equation}
  \mathrm{logit}_3(\tilde{x}_i)=F(\tilde{x}_i),
\end{equation}
其中 $F(\tilde{x}_i)$ 为梯度提升累积输出。正类概率由 Sigmoid 函数转换：
\begin{equation}
  P(Y_3=1\mid \tilde{x}_i)=\frac{1}{1+\exp(-\mathrm{logit}_3(\tilde{x}_i))}.
\end{equation}

模型采用二分类交叉熵损失函数：
\begin{equation}
  L_3=-\frac{1}{n}\sum_{i=1}^{n}
  \bigl[y_{3i}\log P(Y_3=1\mid \tilde{x}_i)
  +(1-y_{3i})\log(1-P(Y_3=1\mid \tilde{x}_i))\bigr].
\end{equation}

在 OOF 预测上搜索最佳分类阈值：
\begin{equation}
  t_3^{*}=\arg\max_{t} ACC_{\mathrm{OOF}}(t),\quad t\in[0.30,0.70].
\end{equation}
最终预测为：
\begin{equation}
  \hat{Y}_3=
  \begin{cases}
    1, & P(Y_3=1\mid \tilde{x}_i)\geq t_3^{*},\\
    0, & P(Y_3=1\mid \tilde{x}_i)<t_3^{*}.
  \end{cases}
\end{equation}

CatBoost 超参数设置：迭代数 500，树深度 6，学习率 0.05，L2 正则化系数 5.0。在 OOF 交叉验证过程中启用早停（early stopping，最多 80 轮无提升则停止），每折的早停轮数中位数用于最终重训的迭代数。

\subsection{问题三结果分析}

三种配置的 OOF 交叉验证结果如表 \ref{tab:task3-compare} 所示。

\begin{table}[H]
  \centering
  \caption{问题三 OOF 性能对比}
  \label{tab:task3-compare}
  \begin{tabular}{lcccc}
    \toprule
    配置 & 特征数 & OOF 准确率 & ROC-AUC & 最优阈值 \\
    \midrule
    A（官方特征） & 5  & 0.7727 & 0.8056 & 0.500 \\
    B（安全扩展） & 18 & 0.8611 & 0.9056 & 0.445 \\
    C（安全+机制） & 24 & 0.8614 & 0.8967 & 0.500 \\
    \bottomrule
  \end{tabular}
\end{table}

配置 B 和 C 的 OOF 准确率相近，均显著优于官方特征。最终选定整体 OOF 准确率最高的配置 C，在 20\% 预留验证集上进行独立评估，得到 ACC3 = 0.8521，ROC-AUC = 0.9015。特征重要性分析显示，环保意识得分、技术接受度、电动车知识和年收入是影响里程焦虑最关键的变量，说明里程焦虑不仅是出行距离的函数，更取决于用户对电动汽车技术的认知水平和接受程度。

\subsection{本章小结}

本章通过多配置特征扩展与 OOF 阈值搜索，建立了 CatBoost 二分类模型用于里程焦虑敏感性识别。实验表明，扩展特征显著提升了模型性能，机制特征的加入进一步带来微幅提升。验证准确率达到 0.8521，相比多数类基线（约 50.8\%）提升了 34.4 个百分点。

\section{综合结果与策略建议}

\subsection{综合结果}

本文三个问题的初步验证结果如表 \ref{tab:summary-results} 所示。

\begin{table}[H]
  \centering
  \caption{三个问题的模型结果汇总}
  \label{tab:summary-results}
  \begin{tabular}{llll}
    \toprule
    问题 & 目标变量 & 主模型 & 验证准确率 \\
    \midrule
    问题一 & 电动汽车采用意愿 & CatBoost三分类 & 0.8592 \\
    问题二 & 家庭充电条件 & CatBoost二分类+阈值搜索 & 0.6498 \\
    问题三 & 里程焦虑敏感性 & CatBoost二分类+OOF阈值搜索 & 0.8521 \\
    \bottomrule
  \end{tabular}
\end{table}

根据题目给定权重，初步模型的综合成绩估计为
\begin{equation}
  Score=100\times(0.2\times0.8592+0.4\times0.6498+0.4\times0.8521)\approx77.26.
\end{equation}

\subsection{策略建议}

根据模型结果，本文提出以下建议：

\begin{enumerate}
  \item 对高采用意愿用户，应重点提供购车补贴、置换优惠和充电权益，促进其尽快转化为实际购买者。
  \item 对中等采用意愿用户，应重点降低其信息不确定性，通过续航说明、真实用车成本对比和试驾体验提升其信心。
  \item 对高里程焦虑用户，应加强公共充电网络信息展示，提供路线规划、充电站推荐和电池质保服务。
  \item 对家庭充电条件不足用户，应推动社区、商圈和办公区公共充电设施建设，降低家庭充电不可得带来的采用阻力。
\end{enumerate}

\section{结论}

本文基于 50000 条电动汽车消费转型数据，围绕电动汽车采用意愿、家庭充电设施安装潜力和里程焦虑敏感性三个问题建立分类预测模型。首先，本文统一了三个问题的符号体系和分类建模框架，并对数据进行缺失值处理、类别变量编码和数值变量标准化。其次，针对问题一建立 CatBoost 三分类模型，针对问题二建立 CatBoost 二分类阈值搜索模型，针对问题三建立 CatBoost 二分类 OOF 阈值搜索模型。初步实验表明，问题一和问题三模型具有较好的预测效果，问题二仍存在较大优化空间。

研究结果表明，电动汽车消费转型是经济因素、基础设施条件、技术认知和心理顾虑共同作用的结果。未来可进一步引入 LightGBM 对照模型、MMoE 多任务学习和 FT-Transformer 深度表格模型，提高模型在家庭充电条件预测任务上的表现，从而提升整体竞赛成绩。

\printbibliography[title={参考文献}]

\appendix

\section{附录}

本文实验代码主要包括数据检查脚本、基线模型脚本和详细结果输出脚本。后续提交时，可将核心训练与预测代码整理为独立程序，并按题目要求生成预测结果 CSV 文件。

\end{document}
