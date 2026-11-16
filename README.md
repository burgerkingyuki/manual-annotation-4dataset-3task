# 引文内容分析：三项标注任务

## 一、引文功能（Citation Function）

| 功能类别 | 描述 | 示例 |
| --- | --- | --- |
| 背景 | 被引文献为施引文献提供研究背景、理论依据或相关领域的已有进展，帮助说明研究问题的来源、研究主题的重要性或相关工作的总体情况，通常用于交代研究背景、概述已有成果或指出领域现状。 | It has been used in syntax-based SMT to model the translation of one NL to another REFLAB. A derivation for a SCFG gives rise to multiple isomorphic parse trees. |
| 使用 | 施引文献直接采用、借鉴或调用被引文献提出的理论、方法、数据集、工具、模型、系统或实验方案，并将其用于自身的研究、分析或实现，此类引用通常体现出较明确的应用或依赖关系。 | Examples include Markov decompositions in M3N REFLAB and dependency-based factorization for MIRA REFLAB. In principle, both methods could be used in our framework. |
| 扩展 | 施引文献以被引文献的理论、方法、数据、工具或系统为基础，进一步对其进行改进、补充、推广或拓展，并形成新的研究内容或成果。此类引用不仅表明施引文献采用了已有工作，还体现出在其基础上有所发展。 | Interjection word n is used as a feature. negation: We manually created a negation list extended by word clusters from REFLAB. A negation word is represented by spellings such as not, n0t, and naht. |
| 对比 | 施引文献将自身研究与被引文献进行比较，讨论二者在研究方法、理论观点、数据、实验结果或适用范围等方面的相似之处与差异，比较可以是明确的，也可以通过上下文隐含表达。 | Our approach outperforms the results by REFLAB and REFLAB. |

## 二、引文对象（Citation Object）

| 对象类别 | 描述 | 示例 |
| --- | --- | --- |
| 研究问题 | 指引文上下文中提及的被引文献所关注或试图解决的研究主题、科学问题或具体任务。 | The effectiveness of these features for [#recognition of discourse relations#] has been previously shown by REFLAB. |
| 研究方法 | 指引文上下文中提及的被引文献用于开展研究或解决问题的方法，包括理论框架、算法模型、实验设计及分析流程等。 | Furthermore, our PROTO+SIM+BOUND model comes close to the [#supervised HMM#] accuracy of 74.4% reported in REFLAB. |
| 数据 | 指引文上下文中提及的被引文献使用、构建或发布的数据资源，包括数据集、语料库、样本及数据集合等。 | In this paper we use the [@AS Internet dataset ITDK0304] collected by CAIDA REFLAB. |
| 其他 | 指不属于研究问题、研究方法或数据，但在引文上下文中被明确提及的其他引用对象，例如评价指标、工具、软件、平台或系统等。 | We compare our extraction algorithm to the [#CATiB schema#] REFLAB. |

## 三、引文情感（Citation Sentiment）

| 情感类别 | 描述 | 示例 |
| --- | --- | --- |
| 正面 | 对被引文献或其研究成果表达肯定、认可或积极评价。 | Dependency relations have been shown to be effective for various sentiment prediction tasks REFLAB REFLAB REFLAB REFLAB. |
| 负面 | 对被引文献或其研究成果表达质疑、批评或否定。 | Many past proposals focused on rules that used word frequency, length, or some combination thereof as a proxy for word complexity REFLAB REFLAB REFLAB, but these features do not correlate perfectly with word complexity for L2 learners. |
| 中性 | 仅对被引文献进行客观陈述、介绍或引用，且上下文中没有明确的肯定、批评或其他情感倾向。 | Airport: user reviews of airports from Skytrax, each annotated with an overall rating at document level REFLAB. |
