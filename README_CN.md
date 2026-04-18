# PyPTO-Lib: 原语张量函数库

[![CI](https://github.com/hw-native-sys/pypto-lib/actions/workflows/ci.yml/badge.svg)](https://github.com/hw-native-sys/pypto-lib/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: CANN Open Software](https://img.shields.io/badge/license-CANN%20OSL%202.0-green.svg)](LICENSE)

> **[English README](README.md)**

---

## 1. 目的

**PyPTO-Lib** 是一个基于 **pypto** 子目录中定义的编程框架构建的**原语张量函数**库。它提供了一组固定的、低层级的、硬件无关的**张量级**操作，作为高层内核和模型图的基础构建块——类似于 **PyTorch ATen** 原语函数在 PyTorch 技术栈中的角色。

**库的定位。** 在**分片（Tile）级别**定义新的函数集几乎没有意义：分片级操作本质上就是 PTO-ISA 指令。库的价值在于**张量级别**：原语是对**张量**的操作（例如 `max(x, axis)`、`exp(x)`、`sum(x, axis)`）。**编译器**负责将这些张量操作分片为分片上的循环，并将每个分片降低到 PTO-ISA。因此 PyPTO-Lib 定义的是**张量级**原语集；分片和 PTO-ISA 是该集合的**降低**，而不是在 PTO-ISA 之上的第二层"分片级原语"。

本文档描述了构建该库的**方法**及其必须实现的**三个核心目标**：

1. **分片和转换为 PTO-ISA**
   **编译器**将张量操作分片为更小的子张量，并将每个分片操作降低为 PTO-ISA。由于 **Tensor** 和 **Tile** 是两种不同的类型，降低过程使用 **`cast_tensor_to_tile`** 和 **`cast_tile_to_tensor`** 进行仅视图转换（无数据移动）。在此阶段**不插入 TLOAD**；核内边界和数据移动由后端在后续阶段决定。

2. **尾块和填充**
   处理分片产生的**尾块**和**填充**，使降低后的代码对任意张量形状都是正确的。

3. **跨组合函数的融合**
   在**一个**组合函数**内部**组合原语（例如 softmax = 一个循环，循环体中包含 max、sub、exp、sum、div）本质上是**手动融合**：用户（或组合函数作者）已经将多个分片操作放入了同一个循环体中。**剩余的挑战**是**如何跨多个组合函数融合循环**——例如当用户编写 `relu(softmax(x))` 时，两个独立的组合函数各有自己的分片循环——使编译器能将它们合并为一个更大循环体的循环。表示和编译器必须支持**跨组合函数的循环融合**，而不仅仅是一个组合函数内的多个分片操作。

---

## 2. Tensor 与 Tile：无数据移动的类型转换

在 pypto 框架中，**Tensor**（N维逻辑张量，例如在 DDR 或全局内存中）和 **Tile**（统一缓冲区/本地内存中的硬件感知块）是**两种不同的类**。库必须在不确定数据实际何时移动的情况下桥接它们。

### 2.1 转换原语

- **`cast_tensor_to_tile(Tensor, offsets, sizes)`**
  生成一个 **Tile**，表示由 `offsets` 和 `sizes` 标识的 Tensor 区域上的**视图**。无数据拷贝：Tile 是该区域上的逻辑描述符（形状、步幅、基指针/视图）。语义为"此分片就是此子张量"；降低为实际的 TLOAD/TSTORE 推迟到编译器选择核内边界时。

- **`cast_tile_to_tensor(Tile)`**
  生成一个 **Tensor**，表示与 Tile 相同逻辑区域上的**视图**。同样无数据移动：它是用于类型兼容性的逆向视图（例如当原语期望 Tensor 但生产者是 Tile 时，或用于链回张量级操作时）。

在**库层面**，我们**不**插入 TLOAD（或类似）指令。库表达的是：
- **分片**：每个操作作用于哪些子张量区域（Tiles）。
- **PTO-ISA 调用**：在这些 Tiles 上的 PTO-ISA 操作序列（逐元素、归约等）。

**数据移动**（何时从 Tensor 内存加载到分片缓冲区，何时存回）是**后续编译关注点**，在编译器决定核内函数边界和放置之后。

---

## 3. 设计原则

### 3.1 框架基础（pypto）

库完全构建在 **pypto** 子目录中的编程框架之上，该框架提供：

- **多级 IR**：张量级和 Tile/Block 级中间表示（参见 `pypto/docs/dev/`、`pypto/include/pypto/ir/`、`pypto/src/ir/`）。
- **IR 构建器和操作**：张量操作、块操作和同步操作注册在 `pypto/src/ir/op/`（例如 `tensor_ops/`、`block_ops/`、`sync_ops/`）。
- **代码生成**：PTO 代码生成器从 pypto IR 生成 PTO-ISA 方言 MLIR（`pypto/include/pypto/codegen/pto/`、`pypto/docs/dev/12-pto_codegen.md`）。
- **后端抽象**：后端注册表和平台特定后端（例如 910B_PTO、910B_CCE），不将原语*语义*绑定到特定执行模型。
- **Python 前端**：`pypto/python/pypto/` 中的语言层和绑定，用于构建和编译程序。

PyPTO-Lib 中的原语表示为 **pypto IR 程序**（张量级和/或 Block 级），然后通过此管线编译。框架不要求库确定"核内"与"编排"；该划分是后端/实现细节。

### 3.2 PTO-ISA 和 ptoas

生成代码的**指令集**和**汇编格式**由 **ptoas**（PTO 汇编器和优化器，位于 **ptoas** 子目录）定义：

- **PTO 方言**：`ptoas/include/PTO/IR/` 中的操作和类型（例如 `PTOOps.td`、`PTOTypeDefs.td`）定义 PTO-ISA 操作（例如 `pto.make_tensor_view`、`pto.alloc_tile`、load/store、逐元素、归约、同步）。
- **降低和 Pass**：`ptoas/lib/PTO/Transforms/` 包含优化和降低 PTO IR 的 Pass（例如同步插入、内存规划），最终到使用 pto-isa C++ 库的代码。
- **工具链**：`ptoas` 工具链消费 `.pto`（PTO 字节码）并生成目标设备的可执行文件或产物。

PyPTO-Lib 原语通过以下方式**降低为 PTO-ISA 指令序列**：

1. **PyPTO IR → PTO-ISA MLIR**：使用 pypto 的 PTO 代码生成器（生成 ptoas 理解的 PTO 方言）。
2. **PTO-ISA MLIR → 二进制**：使用 ptoas 进行解析、优化和代码生成。

因此库**以 PTO-ISA 语义定义**（如 ptoas 中所定义），同时通过 pypto 框架**构建和绑定**。

### 3.3 原语契约中无核内与编排之分

**原语库的 API 和语义**不指定：

- 哪些工作在 **AICore**（核内计算）上运行，哪些在 **AICPU**（编排/调度）上运行。
- load/store、同步和计算如何映射到特定硬件单元。

该映射由以下负责：

- **pypto 后端**和**代码生成器**（例如 910B_PTO、910B_CCE），
- **ptoas** 降低和目标特定代码生成器，
- 以及执行结果二进制文件的**运行时**（例如 pto-rt2/simpler）。

原语是**语义构建块**（例如"逐元素加法"、"矩阵乘法"、"按轴求和"）；后端和运行时决定如何将它们放置在核心和流水线上。

### 3.4 绑定到 PyPTO 前端

原语**绑定到 pypto 前端**，使得：

- 用户可以通过 pypto Python API（例如 `pypto.language`、`pypto.ir`）和 IR 构建器**组合**它们。
- **编译**使用 pypto 的 Pass 和 PTO 代码生成器，然后使用 ptoas 进行 PTO-ISA 级优化和代码生成。
- **注册**遵循 pypto 的算子模型（例如 `REGISTER_OP`、张量/块/同步操作类别），使原语作为 IR 和语言层中的一等操作出现。

因此：**PyPTO-Lib = 一组精选的原语张量函数，实现为 pypto IR（因此也是 PTO-ISA），通过 pypto 前端暴露，不固定核内与编排之分。**

### 3.5 尾块和填充

分片将张量划分为完整分片和可能的**尾部**（不满一个完整分片的剩余维度）。库必须：

- **检测尾块**：当维度不能被分片大小整除时，该维度上的最后一个分片具有较小的逻辑大小。
- **填充或掩码**：(a) 将尾分片填充到完整分片形状，并在 PTO-ISA 中使用**掩码**使填充元素不影响结果（例如归约、max），或 (b) 为尾部生成单独的代码路径，使用较小的分片且无填充。库暴露一致的接口（例如 PTO-ISA 术语中的 `valid_row` / `valid_col`），以便后端降低到适当的形式。
- **正确性**：所有原语（归约、逐元素等）必须遵守尾部语义，使组合函数（例如行上的 softmax）在边界处保持正确。

### 3.6 融合：组合内部 vs 跨组合

- **在单个组合函数内部**：将多个分片（PTO-ISA）操作放入同一循环体——例如 softmax 作为一个循环，循环体为 (max, sub, exp, sum, div)——是**手动融合**。组合函数的作者已经将分片操作序列融合到一个循环中。不需要单独的"融合 Pass"；价值在于拥有清晰的张量级原语集和为每个组合函数生成一个分片循环的降低。

- **跨多个组合函数**：**真正的挑战**是**跨组合函数融合循环**。用户可能编写 `z = relu(softmax(x))`：两个不同的组合函数，各自降低为自己的分片循环（softmax 的分片循环；relu 的分片循环）。没有融合时，生成的代码是"对张量进行完整的 softmax 遍历"然后"对张量进行完整的 relu 遍历"，中间结果被物化。要融合，编译器必须**合并两个循环**为一个：对每个分片，执行（该分片的 softmax 体；然后该分片的 relu 体），使分片在两者之间不离开快速内存。这需要：
  - **表示**：组合函数必须以其**循环结构**（分片边界、迭代空间）和**体**对编译器可见的形式表达。
  - **数据流**：编译器必须看到第一个组合函数（softmax）的输出是第二个（relu）在同一逻辑张量/分片上的唯一输入。
  - **迭代空间对齐**：两个循环必须遍历相同的分片网格（相同形状、相同分片策略）。
  - **合法性**：两个循环之间没有对第一个组合函数完整张量输出的中间使用；没有会因重排序而违反的依赖。

---

## 4. 核内作用域（Incore Scope）

**核内作用域**是用户指定**编排（如主机/AICPU）与核内计算（如 AICore）之间边界**的方式，不改变逻辑算法。通过在 Python 程序源代码中插入**核内作用域指令**，用户可以定义核内作用域的边界，并可**轻松调整**（例如将几行代码移入或移出作用域）以调优放置和数据移动。

### 4.1 核内作用域的含义

- 核内作用域定义一个**匿名核内函数**：一个代码区域，将被编译为单独的**核内**内核（例如在 AICore 上运行），**无需用户显式指定函数参数**。编译器根据变量在作用域边界两侧的使用方式推导参数（见下文）。

- **定义和调用在同一处**：核内作用域**不仅**定义函数，还在**父函数**中作用域出现的**确切位置**定义对该函数的**引用（调用）**。因此父代码（编排）运行到作用域处，然后**调用**生成的核内函数（带推导的参数），然后继续。无需单独的"在别处定义，在此处调用"；作用域既是定义也是调用点。

- **命名**：编译器自动为匿名核内函数生成**有区分度的名称**，使用**父函数名称作为前缀**以保持名称可读和可调试（例如 `my_kernel_incore_0`、`my_kernel_incore_1`）。

### 4.2 编译器推导的参数

编译器使用以下规则**自动推导**匿名核内函数的参数：

| 角色 | 规则 | 含义 |
|------|------|------|
| **输入（Input）** | 在作用域**外部**定义，在作用域**内部引用但未修改**（只读） | 作为核内函数的**输入**参数传递 |
| **输入输出（Inout）** | 在作用域**外部**定义，在作用域**内部被修改** | 作为**输入输出**参数传递（例如按引用） |
| **输出（Output）** | 在作用域**外部**定义；父函数**未赋值**；核内作用域**赋值（写入）**；父函数在作用域后**读取** | 作为**输出**参数按引用传递。符号的**内存空间**由**运行时在核内函数被调用（提交）时分配** |

### 4.3 如何选择 incore_scope

核心**经验法则**：**核内作用域内的中间数据使用量不得超过处理器核心内部 SRAM 缓冲区的大小**。核内函数的工作空间（输入、输出以及作用域内使用的所有临时变量）需要**装入核上 SRAM**。否则数据将**溢出到全局内存**，导致**严重的性能损失**。

### 4.4 示例：核内作用域定义及分析后的等价形式

**用户源代码：**

```python
def my_kernel(x: Tensor, y: Tensor) -> Tensor:
    n, c = x.shape[0], x.shape[1]
    tmp: Tensor   # 外部定义，父函数未赋值；核内作用域内写入；作用域后读取
    with incore_scope():
        for i in range(n):
            for j in range(c):
                tmp[i, j] = x[i, j] + y[i, j]
    result = reduce_sum(tmp, axis=1)
    return result
```

**编译器分析后的等价形式：**

```python
# 编译器生成的核内函数
def my_kernel_incore_0(
    x: Tensor,      # 输入：外部定义，内部只读
    y: Tensor,      # 输入：外部定义，内部只读
    tmp: Tensor,    # 输出：按引用传递；运行时分配内存
) -> None:
    n, c = x.shape[0], x.shape[1]
    for i in range(n):
        for j in range(c):
            tmp[i, j] = x[i, j] + y[i, j]

# 父函数带显式调用
def my_kernel(x: Tensor, y: Tensor) -> Tensor:
    n, c = x.shape[0], x.shape[1]
    tmp: Tensor
    my_kernel_incore_0(x, y, tmp)
    result = reduce_sum(tmp, axis=1)
    return result
```

### 4.5 总结

- **核内作用域指令**在源代码中标记一个区域，该区域成为**匿名核内函数**加上在该位置的**调用**。
- **参数被推导**：**输入**（外部定义，内部只读）、**输入输出**（外部定义，内部被修改）、**输出**（外部定义，父函数未赋值，核内作用域内写入，父函数在作用域后读取；按引用传递；运行时在核内函数被调用/提交时分配内存）。
- 编译器生成**可读名称**（父函数名作前缀）和**显式**函数参数及调用点。

---

## 5. 原语集：ATen 级别的范围（张量级）

PyPTO-Lib 中的原语函数集是**张量级**的，范围类似于 **PyTorch ATen**：API 操作张量（形状、轴、数据类型）。编译器随后对其进行分片并降低到 PTO-ISA；PTO-ISA 之上没有单独的"分片级原语集"。名册包括：

- **逐元素二元操作**：add、sub、mul、div（适用时支持广播）。
- **逐元素一元操作**：sqrt、exp、log、neg 等。
- **归约**：sum、max、min（按轴，可选 keepdim）。
- **线性代数**：matmul、batch matmul（及相关操作）。
- **内存和索引**：张量与分片缓冲区之间的 load/store；索引、切片、类视图操作。
- **类型和布局**：cast、reshape、broadcast（作为后端可映射到 PTO-ISA 的原语）。
- **同步/控制**：IR 暴露的同步原语（例如 sync_src、sync_dst、屏障），仍不指定它们是核内还是编排实现。

**PyPTO-Lib 是原语的封闭集**，融合操作和完整模型图构建在其之上，就像 ATen 之于 PyTorch。

---

## 6. 示例：从张量级原语构建 Softmax

### 6.1 使用的张量级原语

Softmax 使用以下原语在**张量级别**表达：

- **max**(x, axis, keepdim) — 归约。
- **sub**(x, y) — 逐元素。
- **exp**(x) — 逐元素。
- **sum**(x, axis, keepdim) — 归约。
- **div**(x, y) — 逐元素。

公式（对最后一个轴）：**softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))**。

### 6.2 Softmax 作为单个组合函数（一个循环 = 手动融合）

在**张量级别**，softmax 是一个按序列**调用**这些原语的单个组合函数。**编译器**将其降低为**一个**分片循环，循环体为降低后的序列（max → sub → exp → sum → div）。"一个循环中的多个分片操作"是**构造性的**：组合函数被定义为一个逻辑函数，降低产生一个循环。这实际上是分片序列到一个循环中的**手动融合**。

### 6.3 真正的挑战：跨多个组合函数的循环融合

当用户编写**两个（或更多）组合函数**序列时，**困难的**问题出现了，例如：

- `y = relu(softmax(x))`
- 或 `z = add(softmax(x), bias)`

编译器此时有：

- **组合函数 1（softmax）**：一个分片循环。
- **组合函数 2（relu 或 add）**：另一个分片循环。

没有融合时，代码是：**loop_softmax**（遍历所有分片）；然后 **loop_relu**（遍历所有分片）。完整的 softmax 输出在两者之间被物化。为避免这种情况，编译器必须**融合两个循环**为一个。

### 6.4 简化跨组合融合的策略

两种互补策略：

#### 策略 1：完全循环展开

**思路：** 完全展开两个分片循环，使**所有循环索引和分片描述符变为常量**。

**优点：**
- **生产者/消费者匹配变得简单**：每个展开的迭代索引 `i`，softmax 在分片 `i` 的输出和 relu 在分片 `i` 的输入是显式的。
- **无符号推理**：每对 (softmax_i, relu_i) 是具体实例。

**代价：** 代码大小随分片数量线性增长；仅适用于小分片数量。

#### 策略 2：2 的幂次谓词展开（无完全展开）

**思路：** 不完全展开。而是将循环改写为**二进制展开**：一组范围为 **2 的幂次**的循环嵌套，带有**谓词**使仅"使用的"迭代执行。

**优点：**
- **统一的块形状**：展开形式中的所有块具有相同的 2 的幂次范围。
- **有界展开**：只有对数数量的"块大小"出现。
- **每块的生产者/消费者**匹配简单。

**策略比较总结**

| 方面 | 完全展开 | 2 的幂次谓词展开 |
|------|---------|----------------|
| **循环变换** | 展开 Pass：用 N 份带常量索引的体替换循环 | 分解为 2 的幂次段 + 谓词 |
| **符号** | 所有索引和分片描述符变为常量 | 块基地址和大小 (2^k) 是关键符号 |
| **代码大小** | 随分片数量线性增长 | 随范围对数增长；可扩展 |
| **融合 Pass** | 展开后：按索引匹配和合并 | 匹配等形状块，验证数据流，合并块体 |

---

## 7. 构建方法总结

1. **在 pypto IR 中定义原语** — 使用现有操作集和注册机制实现每个原语。
2. **降低到 PTO-ISA** — 使用 pypto 的 PTO 代码生成器生成 PTO 方言 MLIR。
3. **用 ptoas 编译** — 将生成的 PTO-ISA 输入 ptoas 工具链进行优化和目标代码生成。
4. **通过 pypto 前端暴露** — 将原语集绑定到 pypto Python API 和语言层。
5. **保持语义后端无关** — 不将"核内"与"编排"固化到原语库契约中。

---

## 8. 与其他仓库的关系

| 组件 | 相对于 PyPTO-Lib 的角色 |
|------|------------------------|
| **pypto** | 编程框架：IR、操作、代码生成、后端注册表、Python 前端。PyPTO-Lib 是**构建在此框架上的库**并绑定到此前端。 |
| **ptoas** | 定义和实现 **PTO-ISA**（方言、操作、Pass、降低）。PyPTO-Lib 原语**降低到 PTO-ISA**并由 ptoas 编译。 |
| **pto-rt2**（simpler） | 在 Ascend（AICPU + AICore）上执行编译后的任务图的运行时。消费 pypto + ptoas 产生的二进制文件；**不定义**原语库。 |

---

## 9. 总结

**PyPTO-Lib** 是**张量级**原语库（不是分片级的）：在分片级别定义另一组函数集对 PTO-ISA 几乎没有附加价值。原语是**张量**操作（max、exp、sum、add、div 等）；**编译器**对其进行分片并降低到 PTO-ISA。

**三个目标：**

1. **分片和 PTO-ISA**：编译器对张量操作进行分片并降低到 PTO-ISA；**cast_tensor_to_tile** / **cast_tile_to_tensor** 提供仅视图的 Tensor↔Tile 转换；**TLOAD** 在核内边界确定前不插入。
2. **尾块和填充**：降低过程处理不可整除的维度和填充以确保行为正确。
3. **跨组合循环融合**：在**一个**组合函数中放置多个分片操作（例如 softmax）是**手动融合**。**剩余挑战**是**跨多个组合函数融合循环**（例如 `relu(softmax(x))` → 一个循环，每个分片先做 softmax 再做 relu）。

**核内作用域**（第 4 节）：用户在 Python 源代码中插入**核内作用域指令**以标记编排和核内计算之间的边界。

---

## 10. 集群内函数组（In-cluster-function-group）

本节描述用于表达在**集群**（本地互连的核心组）上运行的计算的前端语言特性。

### 10.1 集群作为虚拟张量

**集群**表示为一个虚拟张量（或标量变量），对应于一组本地互连的核心。在 **A5 Ascend** 处理器上，一个集群由 **2 个 AIV** 和 **1 个 AIC** 核心组成。

### 10.2 集群内函数和通信

**集群内函数**是一组使用**本地互连通道**相互通信的函数，抽象为 **push** 和 **pop** 操作。在该组内，数据通信表示为核内函数内的 **TPUSH** 和 **TPOP** 操作，而非 **TSTORE** 和 **TLOAD**。

### 10.3 作用域语法：allocate_cluster

- **`allocate_cluster`** — 调用 pto 运行时分配一个可用的处理器集群，返回标识已分配集群的 **`clusterID`**。
- **阻塞语义** — 如果没有空闲集群，运行时**阻塞**编排直到有空闲集群可用。
- **clusterID 作为输入** — clusterID 是集群内函数组中所有函数的**输入参数**，记录在 pto 运行时任务描述符上。
- **作用域结束** — 程序不显式释放集群；当 clusterID 张量被运行时释放时，pto 运行时**自动**将该集群返回可用池。

### 10.4 核内参数类型：PIPE_IN 和 PIPE_OUT

- **PIPE_IN** 和 **PIPE_OUT** 表示使用**本地互连管道**传递数据的变量，而非全局内存张量。
- 运行时**不**为 PIPE_IN/PIPE_OUT 参数分配全局内存。
- **排空不变量**：程序员必须确保每个集群内函数组在作用域结束前**完全排空**互连管道。
- **一个生产者，多个消费者**：必须表示为**多个单独的 PIPE_OUT** 变量。

---

## 11. block_incore 函数

本节添加用于表达 **block_incore** 函数的语法：以 **SPMD**（单程序多数据）方式执行的核内函数。

### 11.1 调用参数：blockdim 和 block_id

- **blockdim** — 块的**总数**。函数每个块调用一次，共有 **blockdim** 个并发调用。
- **block_id** — 此次调用的**块索引**，范围 `0 .. blockdim-1`。

### 11.2 与集群内函数组配合使用

当 **block_incore** 函数与集群内函数组**一起使用**时，分配的集群数量必须**等于 blockdim**。

### 11.3 优势和编排模式

- **任务压缩和运行时开销** — 通过调度少量 block_incore 任务而非大量细粒度任务，**降低** pto 运行时的开销。
- **PyTorch 急切执行模式** — block_incore 函数也可在 **PyTorch 急切执行模式**中编排，直接从 Python 启动 SPMD 内核，无需使用 pto 运行时。

---

## 12. 项目结构

```
pypto-lib/
├── README.md                     # 英文文档
├── README_CN.md                  # 本文档（中文版）
├── LICENSE
├── ruff.toml                     # Ruff 代码检查配置（Python 3.10+，行宽 110）
├── .pre-commit-config.yaml       # Pre-commit 钩子：头部检查、纯英文检查、ruff
├── .github/
│   ├── workflows/ci.yml          # CI：pre-commit、模拟器测试、设备端（a2a3）测试
│   └── ISSUE_TEMPLATE/           # Bug 报告、功能请求、文档模板
├── golden/                       # Golden 测试基础设施
│   ├── __init__.py               # 导出：TensorSpec、validate_golden、RunConfig、run
│   ├── tensor_spec.py            # TensorSpec 数据类，用于张量描述
│   ├── runner.py                 # 编译 → 执行 → Golden 对比管线
│   └── validation.py             # 基于 torch.allclose 的输出验证
├── examples/
│   ├── beginner/                 # 入门示例
│   │   ├── hello_world.py        # 逐元素加一，行块分片
│   │   └── matmul.py             # 分片矩阵乘法，M/N 块划分
│   ├── intermediate/             # 常用 ML 内核
│   │   ├── softmax.py            # 行级数值稳定 softmax
│   │   ├── layer_norm.py         # 完整 LayerNorm（含 gamma/beta）
│   │   ├── rms_norm.py           # RMSNorm，行+列分片（两遍）
│   │   ├── rope.py               # 旋转位置编码（半向量旋转）
│   │   └── gemm.py               # GEMM，M/N/K 块划分（matmul_acc）
│   └── models/                   # 完整模型层实现
│       ├── qwen3/                # Qwen3-32B：解码、预填充、训练、tilelet 变体
│       ├── deepseek_v3_2/        # DeepSeek V3.2：MLA 解码/预填充（前端/后端拆分）
│       ├── kimi/                 # Kimi K2：MoE 解码，滑动窗口注意力
│       └── milm/                 # MiLM-7B：Llama 风格解码，SwiGLU + GQA
├── tests/
│   ├── golden/                   # Golden 基础设施的单元测试
│   │   ├── conftest.py           # 将仓库根目录加入 sys.path
│   │   ├── test_tensor_spec.py   # TensorSpec.create_tensor() 的测试
│   │   └── test_validation.py    # validate_golden() 的测试
│   └── lint/
│       ├── check_headers.py      # 验证所有 .py 文件的 CANN 许可证头
│       └── check_english_only.py # 确保源文件仅包含 ASCII/英文字符
└── docs/
    ├── para_for.md               # 循环语法：统一的 for/pl.range、parallel、chunk
    ├── pto2_rt.md                # PTO-RT2 运行时设计
    └── pypto-frontend-coding-style.md  # 前端编码规范
```

---

## 13. 前置条件和安装

### 13.1 系统要求

- **Python** >= 3.10
- **PyTorch**（CPU 或 NPU 构建版本）
- **pypto** — 编译器和语言框架（[github.com/hw-native-sys/pypto](https://github.com/hw-native-sys/pypto)）
- **ptoas** — PTO 汇编器和优化器二进制文件（[github.com/hw-native-sys/PTOAS](https://github.com/hw-native-sys/PTOAS)）
- **pto-isa** — PTO-ISA C++ 运行时库（[github.com/hw-native-sys/pto-isa](https://github.com/hw-native-sys/pto-isa)）

对于**设备端**执行（Ascend NPU）：

- Ascend CANN 工具包（例如 8.5.0）
- 可通过 `npu-smi` 访问的 NPU 设备

### 13.2 安装步骤

```bash
# 1. 克隆 pypto-lib
git clone https://github.com/hw-native-sys/pypto-lib.git
cd pypto-lib

# 2. 安装 pypto（编译器 + 运行时）
git clone --recurse-submodules --depth=1 https://github.com/hw-native-sys/pypto.git /tmp/pypto
pip install /tmp/pypto
pip install /tmp/pypto/runtime        # simpler 运行时

# 3. 安装 ptoas 二进制文件（示例：x86_64，v0.25）
curl -L https://github.com/hw-native-sys/PTOAS/releases/download/v0.25/ptoas-bin-x86_64.tar.gz \
  -o /tmp/ptoas.tar.gz
mkdir -p $HOME/ptoas-bin && tar -xzf /tmp/ptoas.tar.gz -C $HOME/ptoas-bin
export PTOAS_ROOT=$HOME/ptoas-bin

# 4. 克隆 pto-isa（固定提交）
git clone https://github.com/hw-native-sys/pto-isa.git $HOME/pto-isa
export PTO_ISA_ROOT=$HOME/pto-isa

# 5. 安装 Python 依赖
pip install torch nanobind
```

---

## 14. 快速开始

### 14.1 Hello World — 逐元素加一

最简单的示例（`examples/beginner/hello_world.py`）将矩阵按行块分片并逐元素加 1：

```python
import pypto.language as pl

ROWS, COLS, ROW_CHUNK = 1024, 512, 128

@pl.program
class HelloWorldProgram:
    @pl.function(type=pl.FunctionType.Opaque)
    def add_one(
        self,
        x: pl.Tensor[[ROWS, COLS], pl.FP32],
        y: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    ) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
            for r in pl.parallel(0, ROWS, 1, chunk=ROW_CHUNK):
                tile_x = pl.slice(x, [1, COLS], [r, 0])
                tile_y = pl.add(tile_x, 1.0)
                y = pl.assemble(y, tile_y, [r, 0])
        return y
```

**关键模式：**

- **`@pl.program`** / **`@pl.function`** — 声明 pypto 程序及其编排函数。
- **`pl.Tensor[[shape], dtype]`** — 张量类型注解；**`pl.Out[...]`** 标记输出张量。
- **`pl.parallel(start, end, step, chunk=C)`** — 基于块的分片并行循环；编译器拆分为块循环（外层）+ 块内循环（内层）。
- **`pl.slice(tensor, sizes, offsets)`** — 从张量中提取一个分片视图。
- **`pl.assemble(dst, tile, offsets)`** — 将分片写回输出张量的指定偏移处。
- **`pl.at(level=..., optimization=...)`** — 指定执行级别和优化策略。

### 14.2 运行示例

```bash
# 模拟器模式（无需 NPU）
python examples/beginner/hello_world.py -p a2a3sim

# 设备端模式（需要 Ascend NPU）
python examples/beginner/hello_world.py -p a2a3 -d 0

# 启用运行时性能分析
python examples/beginner/hello_world.py -p a2a3 -d 0 --runtime-profiling
```

所有示例接受相同的命令行参数：

| 参数 | 说明 |
|------|------|
| `-p`, `--platform` | 目标平台：`a2a3`、`a2a3sim`、`a5`、`a5sim` |
| `-d`, `--device` | 设备 ID（默认：0） |
| `--runtime-profiling` | 启用运行时性能分析输出 |

---

## 15. `pypto.language` API 参考（常用操作）

以下来自 `pypto.language`（导入为 `pl`）的操作在示例中广泛使用，构成 PyPTO-Lib 的**张量级原语集**。

### 15.1 张量构造和视图

| API | 说明 |
|-----|------|
| `pl.Tensor[[shape], dtype]` | 张量类型注解（如 `pl.FP32`、`pl.BF16`） |
| `pl.Out[pl.Tensor[...]]` | 输出张量注解 |
| `pl.slice(tensor, sizes, offsets)` | 在给定偏移处从张量提取分片（视图） |
| `pl.assemble(dst, tile, offsets)` | 在给定偏移处将分片写回张量 |
| `pl.create_tensor(shape, dtype=...)` | 分配新的本地张量 |
| `pl.reshape(tensor, new_shape)` | 重塑张量（视图操作） |

### 15.2 逐元素操作

| API | 说明 |
|-----|------|
| `pl.add(a, b)` | 逐元素加法（张量+张量 或 张量+标量） |
| `pl.sub(a, b)` | 逐元素减法 |
| `pl.mul(a, b)` | 逐元素乘法 |
| `pl.exp(x)` | 逐元素指数函数 |
| `pl.sqrt(x)` | 逐元素平方根 |
| `pl.rsqrt(x)` | 逐元素倒数平方根（1/sqrt） |
| `pl.neg(x)` | 逐元素取反 |
| `pl.recip(x)` | 逐元素倒数（1/x） |

### 15.3 行/列广播操作

| API | 说明 |
|-----|------|
| `pl.row_expand_sub(matrix, row_vec)` | 每行减去一个行向量 |
| `pl.row_expand_mul(matrix, row_vec)` | 每行乘以一个行向量 |
| `pl.row_expand_div(matrix, row_vec)` | 每行除以一个行向量 |
| `pl.col_expand_mul(matrix, col_vec)` | 每列乘以一个列向量 |

### 15.4 归约

| API | 说明 |
|-----|------|
| `pl.row_max(matrix)` | 行级最大值（返回列向量） |
| `pl.row_sum(matrix)` | 行级求和（返回列向量） |

### 15.5 线性代数

| API | 说明 |
|-----|------|
| `pl.matmul(a, b)` | 矩阵乘法：`C = A @ B` |
| `pl.matmul_acc(acc, a, b)` | 累加矩阵乘法：`acc += A @ B` |

### 15.6 循环构造

| API | 说明 |
|-----|------|
| `pl.parallel(start, end, step, chunk=C)` | 基于块的分片并行循环 |
| `pl.range(end)` / `pl.range(start, end)` / `pl.range(start, end, step)` | 顺序循环（类似 Python `range`） |

### 15.7 执行上下文

| API | 说明 |
|-----|------|
| `pl.at(level=..., optimization=...)` | 设置执行级别（`pl.Level.CORE_GROUP`）和优化器（`pl.chunked_loop_optimizer`） |
| `@pl.program` | 类装饰器，声明 pypto 程序 |
| `@pl.function(type=pl.FunctionType.Opaque)` | 方法装饰器，声明编排函数 |

---

## 16. 示例库

### 16.1 入门级

| 示例 | 文件 | 说明 |
|------|------|------|
| **Hello World** | `examples/beginner/hello_world.py` | 逐元素 `y = x + 1`，行块并行循环 |
| **Matmul** | `examples/beginner/matmul.py` | 分片 `C = A @ B`，M/N 并行块划分（无 K 分片） |

### 16.2 中级

| 示例 | 文件 | 说明 |
|------|------|------|
| **Softmax** | `examples/intermediate/softmax.py` | 行级数值稳定 softmax：max → sub → exp → sum → div |
| **LayerNorm** | `examples/intermediate/layer_norm.py` | 完整 `(x - mean) / sqrt(var + eps) * gamma + beta`，仅行分片 |
| **RMSNorm** | `examples/intermediate/rms_norm.py` | 两遍 RMSNorm，行+列分片，适用于大隐藏维度 |
| **RoPE** | `examples/intermediate/rope.py` | 旋转位置编码，半向量 cos/sin 旋转 |
| **GEMM** | `examples/intermediate/gemm.py` | 完整 GEMM，M/N/K 块划分，使用 `pl.matmul` + `pl.matmul_acc` |

**示例：Softmax 内核**（`examples/intermediate/softmax.py`）

```python
for r in pl.parallel(0, rows, row_chunk, chunk=1):
    tile_x = pl.slice(x, [row_chunk, cols], [r, 0])
    row_max = pl.row_max(tile_x)                      # 第 1 步：行级最大值
    shifted = pl.row_expand_sub(tile_x, row_max)       # 第 2 步：x - max(x)
    exp_shifted = pl.exp(shifted)                       # 第 3 步：exp(x - max(x))
    row_sum = pl.row_sum(exp_shifted)                   # 第 4 步：行级求和
    result = pl.row_expand_div(exp_shifted, row_sum)    # 第 5 步：归一化
    y = pl.assemble(y, result, [r, 0])
```

**示例：带 K 分片的 GEMM**（`examples/intermediate/gemm.py`）

```python
for mb in pl.parallel(0, m, m_tile, chunk=m_chunk):
    for nb in pl.parallel(0, n, n_tile, chunk=n_chunk):
        tile_a = pl.slice(a, [m_tile, k_tile], [mb, 0])
        tile_b = pl.slice(b, [k_tile, n_tile], [0, nb])
        acc = pl.matmul(tile_a, tile_b)                 # 第一个 K 分片
        for kb in pl.range(1, k_blocks):                # 剩余 K 分片
            k0 = kb * k_tile
            tile_a_i = pl.slice(a, [m_tile, k_tile], [mb, k0])
            tile_b_i = pl.slice(b, [k_tile, n_tile], [k0, nb])
            acc = pl.matmul_acc(acc, tile_a_i, tile_b_i)
        c = pl.assemble(c, acc, [mb, nb])
```

### 16.3 模型级示例

面向生产级 LLM 的完整单层 Transformer 实现：

| 模型 | 目录 | 亮点 |
|------|------|------|
| **Qwen3-32B** | `examples/models/qwen3/` | 14 个变体：解码、预填充、训练前向+反向、tilelet、逐作用域分解、混合精度 |
| **DeepSeek V3.2** | `examples/models/deepseek_v3_2/` | MLA（多头潜在注意力）前端/后端拆分，通过 `index_topk` 的稀疏注意力 |
| **Kimi K2** | `examples/models/kimi/` | MoE（8 专家 + 1 共享专家），滑动窗口注意力，SwiGLU |
| **MiLM-7B** | `examples/models/milm/` | Llama 风格 GQA + SwiGLU + RoPE，针对移动/边缘部署优化 |

这些模型示例展示了实际的生产模式：**多作用域核内边界**、**K/V 缓存管理**、**MoE 路由**、**两遍归一化**和**融合注意力内核**。

---

## 17. Golden 测试框架

`golden/` 包提供端到端的**编译 → 执行 → 验证**管线，用于将 PyPTO 程序与 PyTorch 参考实现进行对比测试。

### 17.1 核心组件

- **`TensorSpec`**（`golden/tensor_spec.py`） — 描述张量的名称、形状、数据类型、初始化策略及是否为输出：

  ```python
  from golden import TensorSpec
  specs = [
      TensorSpec("x", [512, 256], torch.float32, init_value=torch.randn),
      TensorSpec("y", [512, 256], torch.float32, is_output=True),
  ]
  ```

  支持的 `init_value` 类型：`None`（零初始化）、`int`/`float`（常量填充）、`torch.Tensor`（直接使用）或可调用对象（`torch.randn`、`torch.rand`、`torch.zeros`、`torch.ones`、或自定义函数）。

- **`run()`**（`golden/runner.py`） — 主入口点：

  ```python
  from golden import RunConfig, run

  result = run(
      program=build_softmax_program(),
      tensor_specs=build_tensor_specs(),
      golden_fn=golden_softmax,      # PyTorch 参考函数
      config=RunConfig(rtol=1e-5, atol=1e-5, runtime=dict(platform="a2a3sim")),
  )
  assert result.passed
  ```

  管线流程：通过 `pypto.ir.compile` *编译* → 从规格 *生成输入* → *在设备上执行* → *计算 Golden 参考* → 使用 `torch.allclose` *验证*。

- **`validate_golden()`**（`golden/validation.py`） — 逐元素比较设备输出与 Golden 张量。不匹配时报告计数和前 20 个不匹配元素的实际值与期望值。

- **`RunConfig`**（`golden/runner.py`） — 配置数据类：

  | 字段 | 类型 | 默认值 | 说明 |
  |------|------|--------|------|
  | `rtol` | `float` | `1e-5` | 相对容差 |
  | `atol` | `float` | `1e-5` | 绝对容差 |
  | `compile_only` | `bool` | `False` | 仅编译不执行 |
  | `compile` | `dict` | `{}` | 转发给 `pypto.ir.compile` 的参数 |
  | `runtime` | `dict` | `{}` | 转发给 `pypto.runtime.RunConfig` 的参数 |

### 17.2 示例文件的统一结构

每个示例遵循一致的三函数模式：

```python
def build_<name>_program(...):       # 返回 @pl.program 类
    ...

def build_tensor_specs(...):          # 返回 list[TensorSpec]
    ...

def golden_<name>(tensors):           # PyTorch 参考（原地修改）
    ...

if __name__ == "__main__":
    result = run(program=..., tensor_specs=..., golden_fn=..., config=...)
```

### 17.3 运行测试

```bash
# Golden 基础设施的单元测试
pytest tests/golden/ -v

# 在模拟器中运行所有入门和中级示例
for f in $(find examples/beginner examples/intermediate -name '*.py' | sort); do
    python "$f" -p a2a3sim
done
```

---

## 18. CI 和支持的平台

### 18.1 CI 管线（`.github/workflows/ci.yml`）

CI 在每次 push/PR 到 `main` 时运行三个任务：

| 任务 | 运行器 | 执行内容 |
|------|--------|----------|
| **pre-commit** | `ubuntu-latest` | 许可证头检查、纯英文检查、ruff 代码检查 |
| **sim** | `ubuntu-latest` | 在 **a2a3sim** 和 **a5sim** 模拟器上运行所有入门和中级示例 |
| **a2a3** | 自托管 ARM64 NPU | 在**真实 Ascend NPU** 上运行所有入门和中级示例 |

### 18.2 支持的平台

| 平台 | 说明 | 需要硬件 |
|------|------|----------|
| `a2a3sim` | Ascend 910B 模拟器 | 否 |
| `a5sim` | Ascend 950 模拟器 | 否 |
| `a2a3` | Ascend 910B 设备端 | 是（NPU） |
| `a5` | Ascend 950 设备端 | 是（NPU） |

### 18.3 Pre-commit 钩子

```bash
# 安装 pre-commit 并运行所有检查
pip install pre-commit
pre-commit run --all-files
```

钩子：
- **check-headers** — 验证所有 `.py` 文件的 CANN 开源软件许可证头。
- **check-english-only** — 确保源文件（`.py`）仅包含 ASCII/英文字符。
- **ruff-check** — 运行 ruff 代码检查器，使用 pyflakes 规则（未使用的导入、未定义的名称）。

---

## 19. 贡献指南

1. Fork 仓库并创建功能分支。
2. 遵循 `docs/pypto-frontend-coding-style.md` 中的编码规范。
3. 确保所有 `.py` 文件包含 CANN 许可证头（参见任何现有文件的模板）。
4. 为新示例添加 Golden 测试（遵循 `build_program` / `build_tensor_specs` / `golden_fn` 模式）。
5. 运行 pre-commit 钩子：`pre-commit run --all-files`。
6. 针对模拟器运行示例：`python examples/<path>.py -p a2a3sim`。
7. 向 `main` 提交 Pull Request。

---

## 20. 许可证

PyPTO-Lib 采用 **CANN 开源软件许可协议第 2.0 版** 授权。完整文本参见 [LICENSE](LICENSE)。
