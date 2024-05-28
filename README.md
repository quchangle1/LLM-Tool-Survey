# Survey: Tool Learning with Large Language Models
Recently, tool learning with large language models~(LLMs) has emerged as a promising paradigm for augmenting the capabilities of LLMs to tackle highly complex problems. 

This is the collection of papers related to tool learning with LLMs. These papers are organized according to our survey paper ["Tool Learning with Large Language Models: A Survey"]().

## 📋 Contents
- [Introduction](#🌟-introduction)
- [Paper List](#📄-paper-list)
  - [Why Tool Learning?](#why-tool-learning)
    - [Benefit of Tools](#benefit-of-tools)
    - [Benefit of Tool Learning.](#benefit-of-tool-learning)
  - [How Tool Learning?](#how-tool-learning)
    - [Task Planning](#task-planning)
      - [Tuning-free Methods](#tuning-free-methods)
      - [Tuning-based Methods](#tuning-based-methods)
    - [Tool Selection](#tool-selection)
      - [Retriever-based Tool Selection](#retriever-based-tool-selection)
      - [LLM-based Tool Selection](#llm-based-tool-selection)
    - [Tool Calling](#tool-calling)
      - [Tuning-free Methods](#tuning-free-methods-1)
      - [Tuning-based Methods](#tuning-based-methods-1)
    - [Response Generation](#response-generation)
      - [Direct Insertion Methods](#direct-insertion-methods)
      - [Information Integration Methods](#information-integration-methods)
  - [Benchmarks and Evaluation](#benchmarks-and-evaluation)
    - [Benchmarks](#benchmarks)
    - [Evaluation](#evaluation)
  - [Challenges and Future Directions](#challenges-and-future-directions)


If you find our paper or code useful, please cite the paper:


## 🌟 Introduction



## 📄 Paper List
### Why Tool Learning?
#### Benefit of Tools.
- Knowledge Acquisition.
  - Search Engine
    
    **Internet-Augmented Dialogue Generation**, ACL 2022. [[Paper]](https://arxiv.org/abs/2107.07566)
    
    **WebGPT: Browser-assisted question-answering with human feedback**, Preprint 2021. [[Paper]](https://arxiv.org/abs/2112.09332)
    
    **Internet-augmented language models through few-shot prompting for open-domain question answering**, Preprint 2022. [[Paper]](https://arxiv.org/abs/2203.05115)
    
    **REPLUG: Retrieval-Augmented Black-Box Language Models**, Preprint 2023. [[Paper]](https://arxiv.org/abs/2301.12652)
    
    **Toolformer: Language Models Can Teach Themselves to Use Tools**, NeurIPS 2023. [[Paper]](https://arxiv.org/abs/2302.04761)
    
    **ART: Automatic multi-step reasoning and tool-use for large language models**, Preprint 2023. [[Paper]](https://arxiv.org/abs/2303.09014)
    
    **CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing**, ICLR 2024. [[Paper]](http://arxiv.org/abs/2305.11738)
    
  - Database & Knowledge Graph
    
    **Lamda: Language models for dialog applications**, Preprint 2022. [[Paper]](https://arxiv.org/abs/2201.08239)
    
    **Gorilla: Large Language Model Connected with Massive APIs**, Preprint 2023. [[Paper]](http://arxiv.org/abs/2305.15334)
    
    **ToolkenGPT: Augmenting Frozen Language Models with Massive Tools via Tool Embeddings**, NeurIPS 2023. [[Paper]](https://arxiv.org/abs/2305.11554)
    
    **ToolQA: A Dataset for LLM Question Answering with External Tools**, NeurIPS 2023. [[Paper]](http://arxiv.org/abs/2306.13304)
    
    **Syntax Error-Free and Generalizable Tool Use for LLMs via Finite-State Decoding**, NeurIPS 2023. [[Paper]](https://arxiv.org/abs/2310.07075)
    
    **Middleware for LLMs: Tools are Instrumental for Language Agents in Complex Environments**, Preprint 2024. [[Paper]](https://arxiv.org/abs/2402.14672)
    
  - Weather or Map
    
    **On the Tool Manipulation Capability of Open-source Large Language Models**, NeurIPS 2023. [[Paper]](http://arxiv.org/abs/2305.16504)
    
    **ToolAlpaca: Generalized Tool Learning for Language Models with 3000 Simulated Cases**, Preprint 2023. [[Paper]](http://arxiv.org/abs/2306.05301)
    
    **Tool Learning with Foundation Models**, Preprint 2023. [[Paper]](http://arxiv.org/abs/2304.08354)

- Expertise Enhancement.
  - Mathematical Tools
    
    **Training verifiers to solve math word problems**, Preprint 2021. [[Paper]](https://arxiv.org/abs/2110.14168)
    
    **MRKL Systems: A modular, neuro-symbolic architecture that combines large language models, external knowledge sources and discrete reasoning**, Preprint 2021. [[Paper]](https://arxiv.org/abs/2205.00445)
    
    **Chaining Simultaneous Thoughts for Numerical Reasoning**, EMNLP 2022. [[Paper]](https://arxiv.org/abs/2211.16482)
    
    **Calc-X and Calcformers: Empowering Arithmetical Chain-of-Thought through Interaction with Symbolic Systems**, EMNLP 2023. [[Paper]](http://arxiv.org/abs/2305.15017)
    
    **Solving math word problems by combining language models with symbolic solvers**, NeurIPS 2023. [[Paper]](http://arxiv.org/abs/2304.09102)
    
    **Evaluating and improving tool-augmented computation-intensive math reasoning**, NeurIPS 2023. [[Paper]](http://arxiv.org/abs/2306.02408)
    
    **ToRA: A Tool-Integrated Reasoning Agent for Mathematical Problem Solving**, ICLR 2024. [[Paper]](https://arxiv.org/abs/2309.17452)
    
    **MATHSENSEI: A Tool-Augmented Large Language Model for Mathematical Reasoning**, Preprint 2024. [[Paper]](http://arxiv.org/abs/2402.17231)
    
    **Calc-CMU at SemEval-2024 Task 7: Pre-Calc -- Learning to Use the Calculator Improves Numeracy in Language Models**, NAACL 2024. [[Paper]](https://arxiv.org/abs/2404.14355)
    
  - Python Interpreter
    
    **Pal: Program-aided language models**, ICML 2023. [[Paper]](https://arxiv.org/abs/2211.10435)
    
    **Program of Thoughts Prompting: Disentangling Computation from Reasoning for Numerical Reasoning Tasks**, TMLR 2023. [[Paper]](https://arxiv.org/abs/2211.12588)
    
    **Fact-Checking Complex Claims with Program-Guided Reasoning**, ACL 2023. [[Paper]](http://arxiv.org/abs/2305.12744)
    
    **Chameleon: Plug-and-Play Compositional Reasoning with Large Language Models**, NeurIPS 2023. [[Paper]](http://arxiv.org/abs/2304.09842)
    
    **LeTI: Learning to Generate from Textual Interactions**, NAACL 2024. [[Paper]](https://arxiv.org/abs/2305.10314)
    
    **Mint: Evaluating llms in multi-turn interaction with tools and language feedback**, ICLR 2024. [[Paper]](http://arxiv.org/abs/2309.10691)
    
  - Others
    
    Chemical: **MultiTool-CoT: GPT-3 Can Use Multiple External Tools with Chain of Thought Prompting**, ACL 2023. [[Paper]](http://arxiv.org/abs/2305.16896)
    
    Biomedical: **GeneGPT: Augmenting Large Language Models with Domain Tools for Improved Access to Biomedical Information**, ISMB 2024. [[Paper]](https://arxiv.org/abs/2304.09667)
    
    Financial: **Equipping Language Models with Tool Use Capability for Tabular Data Analysis in Finance**, EACL 2024. [[Paper]](https://arxiv.org/abs/2401.15328)
    
    Medical: **AgentMD: Empowering Language Agents for Risk Prediction with Large-Scale Clinical Tool Learning**, Preprint 2024. [[Paper]](https://arxiv.org/abs/2402.13225)

    Recommendation: **Let Me Do It For You: Towards LLM Empowered Recommendation via Tool Learning**, SIGIR 2024. [[Paper]](https://arxiv.org/abs/2405.15114)

##### Automation and Efficiency.
  - Schedule Tools
    
    **ToolQA: A Dataset for LLM Question Answering with External Tools**, NeurIPS 2023. [[Paper]](http://arxiv.org/abs/2306.13304)

  - Set Reminders
    
    **ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs**, ICLR 2024. [[Paper]](http://arxiv.org/abs/2307.16789)
    
  - Filter Emails
   
    **ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs**, ICLR 2024. [[Paper]](http://arxiv.org/abs/2307.16789)
    
  - Project Management

    **ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs**, ICLR 2024. [[Paper]](http://arxiv.org/abs/2307.16789)
    
  - Online Shopping Assistants
    
    **WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents**, NeurIPS 2022. [[Paper]](https://arxiv.org/abs/2207.01206)


##### Interaction Enhancement.
  - Multi-modal Tools
    
    **Vipergpt: Visual inference via python execution for reasoning**, ICCV 2023. [[Paper]](https://arxiv.org/abs/2303.08128)
    
    **MM-REACT: Prompting ChatGPT for Multimodal Reasoning and Action**, Preprint 2023. [[Paper]](https://arxiv.org/abs/2303.11381)
    
    **InternGPT: Solving Vision-Centric Tasks by Interacting with ChatGPT Beyond Language**, Preprint 2023. [[Paper]](https://arxiv.org/abs/2305.05662)
    
    **AssistGPT: A General Multi-modal Assistant that can Plan, Execute, Inspect, and Learn**, Preprint 2023. [[Paper]](https://arxiv.org/abs/2306.08640)
    
    **CLOVA: A closed-loop visual assistant with tool usage and update**, CVPR 2024. [[Paper]](https://arxiv.org/abs/2312.10908)
    
    **DiffAgent: Fast and Accurate Text-to-Image API Selection with Large Language Model**, CVPR 2024. [[Paper]](http://arxiv.org/abs/2404.01342)
    
    **MLLM-Tool: A Multimodal Large Language Model For Tool Agent Learning**, Preprint 2024. [[Paper]](http://arxiv.org/abs/2401.10727)
    
    **m&m's: A Benchmark to Evaluate Tool-Use for multi-step multi-modal Tasks**, Preprint 2024. [[Paper]](http://arxiv.org/abs/2403.11085)
    
  - Machine Translator
    
    **Toolformer: Language Models Can Teach Themselves to Use Tools**, NeurIPS 2023. [[Paper]](https://arxiv.org/abs/2302.04761)
    
    **Tool Learning with Foundation Models**, Preprint 2023. [[Paper]](http://arxiv.org/abs/2304.08354)
    
  - Natural Language Processing Tools
    
    **HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face**, NeurIPS 2023. [[Paper]](https://arxiv.org/abs/2303.17580)
    
    **GitAgent: Facilitating Autonomous Agent with GitHub by Tool Extension**, Preprint 2023. [[Paper]](https://arxiv.org/abs/2312.17294)

#### Benefit of Tool Learning.

- Enhanced Interpretability and User Trust.
- Improved Robustness and Adaptability.




### How Tool Learning?
#### Task Planning.
- ##### Tuning-free Methods
  
  **Chain-of-Thought Prompting Elicits Reasoning in Large Language Models**, NeurIPS 2022. [[Paper]](https://arxiv.org/abs/2201.11903)
  
  **ReAct: Synergizing Reasoning and Acting in Language Models**, ICLR 2023. [[Paper]](https://arxiv.org/abs/2210.03629)
  
  **ART: Automatic multi-step reasoning and tool-use for large language models**, Preprint 2023. [[Paper]](https://arxiv.org/abs/2303.09014)
  
  **HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face**, NeurIPS 2023. [[Paper]](https://arxiv.org/abs/2303.17580)
  
  **Graph-ToolFormer: To Empower LLMs with Graph Reasoning Ability via Prompt Augmented by ChatGPT**, Preprint 2023. [[Paper]](https://arxiv.org/abs/2304.11116)
  
  **Large Language Models as Tool Makers**, ICLR 2024. [[Paper]](https://arxiv.org/abs/2305.17126)
  
  **CREATOR: Tool Creation for Disentangling Abstract and Concrete Reasoning of Large Language Models**, EMNLP 2023. [[Paper]](https://arxiv.org/abs/2305.14318)
  
  **FacTool: Factuality Detection in Generative AI -- A Tool Augmented Framework for Multi-Task and Multi-Domain Scenarios**, Preprint 2023. [[Paper]](https://arxiv.org/abs/2307.13528)
  
  **TPTU: Large Language Model-based AI Agents for Task Planning and Tool Usage**, Preprint 2023. [[Paper]](https://arxiv.org/abs/2308.03427)
  
  **ToolChain*: Efficient Action Space Navigation in Large Language Models with A* Search**, ICLR 2024. [[Paper]](https://arxiv.org/abs/2310.13227)
  
  **Fortify the Shortest Stave in Attention: Enhancing Context Awareness of Large Language Models for Effective Tool Use**, Preprint 2023. [[Paper]](https://arxiv.org/abs/2312.04455)
  
  **TroVE: Inducing Verifiable and Efficient Toolboxes for Solving Programmatic Tasks**, Preprint 2023. [[Paper]](https://arxiv.org/abs/2401.12869)
  
  **Planning and Editing What You Retrieve for Enhanced Tool Learning**, NAACL 2024. [[Paper]](https://arxiv.org/abs/2404.00450)
  
- ##### Tuning-based Methods
  
  **TaskMatrix.AI: Completing Tasks by Connecting Foundation Models with Millions of APIs**, INTELLIGENT COMPUTING 2024. [[Paper]](https://arxiv.org/abs/2303.16434)
  
  **ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs**, ICLR 2024. [[Paper]](https://arxiv.org/abs/2307.16789)
  
  **Toolink: Linking Toolkit Creation and Using through Chain-of-Solving on Open-Source Model**, Preprint 2023. [[Paper]](https://arxiv.org/abs/2310.05155)
  
  **TPTU-v2: Boosting Task Planning and Tool Usage of Large Language Model-based Agents in Real-world Systems**, ICLR 2024. [[Paper]](https://arxiv.org/abs/2311.11315)
  
  **Navigating Uncertainty: Optimizing API Dependency for Hallucination Reduction in Closed-Book Question Answering**, ECIR 2024. [[Paper]](https://arxiv.org/abs/2401.01780)
  
  **Small LLMs Are Weak Tool Learners: A Multi-LLM Agent**, Preprint 2024. [[Paper]](https://arxiv.org/abs/2401.07324)
  
  **Efficient Tool Use with Chain-of-Abstraction Reasoning**, Preprint 2024. [[Paper]](https://arxiv.org/abs/2401.17464)
  
  **Look Before You Leap: Towards Decision-Aware and Generalizable Tool-Usage for Large Language Models**, Preprint 2024. [[Paper]](https://arxiv.org/abs/2402.16696)

#### Tool Selection.
- ##### Retriever-based Tool Selection
  
  **A statistical interpretation of term specificity and its application in retrieval**, Journal of Documentation 1972. [[Paper]](https://www.emerald.com/insight/content/doi/10.1108/eb026526/full/html)
  
  **The probabilistic relevance framework: BM25 and beyond**, Foundations and Trends in Information Retrieval 2009. [[Paper]](https://dl.acm.org/doi/10.1561/1500000019)
  
  **Sentence-bert: Sentence embeddings using siamese bert-networks**, EMNLP 2019. [[Paper]](https://arxiv.org/abs/1908.10084)
  
  **Approximate nearest neighbor negative contrastive learning for dense text retrieval**, ICLR 2021. [[Paper]](https://arxiv.org/abs/2007.00808)
  
  **Efficiently Teaching an Effective Dense Retriever with Balanced Topic Aware Sampling**, SIGIR 2021. [[Paper]](https://arxiv.org/abs/2104.06967)
  
  **Unsupervised Corpus Aware Language Model Pre-training for Dense Passage Retrieval**, ACL 2022. [[Paper]](https://arxiv.org/abs/2108.05540)
  
  **Unsupervised dense information retrieval with contrastive learning**, Preprint 2021. [[Paper]](https://arxiv.org/abs/2112.09118)
  
  **CRAFT: Customizing LLMs by Creating and Retrieving from Specialized Toolsets**, ICLR 2024. [[Paper]](https://arxiv.org/abs/2309.17428)
  
  **ProTIP: Progressive Tool Retrieval Improves Planning**, Preprint 2023. [[Paper]](https://arxiv.org/abs/2312.10332)
  
  **ToolRerank: Adaptive and Hierarchy-Aware Reranking for Tool Retrieval**, COLING 2024. [[Paper]](https://arxiv.org/abs/2403.06551)
  
  **COLT: Towards Completeness-Oriented Tool Retrieval for Large Language Models**, Preprint 2024. [[Paper]](https://arxiv.org/abs/2405.16089)

- ##### LLM-based Tool Selection
  
  **On the Tool Manipulation Capability of Open-source Large Language Models**, Preprint 2023. [[Paper]](https://arxiv.org/abs/2305.16504)
  
  **Making Language Models Better Tool Learners with Execution Feedback**, NAACL 2024. [[Paper]](https://arxiv.org/abs/2305.13068)
  
  **ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs**, ICLR 2024. [[Paper]](https://arxiv.org/abs/2307.16789)
  
  **Confucius: Iterative Tool Learning from Introspection Feedback by Easy-to-Difficult Curriculum**, AAAI 2024. [[Paper]](https://arxiv.org/abs/2308.14034)
  
  **AnyTool: Self-Reflective, Hierarchical Agents for Large-Scale API Calls**, Preprint 2024. [[Paper]](https://arxiv.org/abs/2402.04253)
  
  **TOOLVERIFIER: Generalization to New Tools via Self-Verification**, Preprint 2024. [[Paper]](https://arxiv.org/abs/2402.14158)
  
  **ToolNet: Connecting Large Language Models with Massive Tools via Tool Graph**, Preprint 2024. [[Paper]](https://arxiv.org/abs/2403.00839)

#### Tool Calling.
- ##### Tuning-free Methods
  
  **RestGPT: Connecting Large Language Models with Real-World RESTful APIs**, Preprint 2023. [[Paper]](https://arxiv.org/abs/2306.06624)
  
  **Reverse Chain: A Generic-Rule for LLMs to Master Multi-API Planning**, Preprint 2023. [[Paper]](http://arxiv.org/abs/2310.04474)
  
  **GEAR: Augmenting Language Models with Generalizable and Efficient Tool Resolution**, EACL 2023. [[Paper]](https://arxiv.org/abs/2307.08775)
  
  **Tool Documentation Enables Zero-Shot Tool-Usage with Large Language Models**, Preprint 2023. [[Paper]](https://arxiv.org/abs/2308.00675)
  
  **ControlLLM: Augment Language Models with Tools by Searching on Graphs**, Preprint 2023. [[Paper]](https://arxiv.org/abs/2310.17796)
  
  **EASYTOOL: Enhancing LLM-based Agents with Concise Tool Instruction**, Preprint 2024. [[Paper]](https://arxiv.org/abs/2401.06201)
  
- ##### Tuning-based Methods
  
  **Gorilla: Large Language Model Connected with Massive APIs**, Preprint 2023. [[Paper]](https://arxiv.org/abs/2305.15334)
  
  **GPT4Tools: Teaching Large Language Model to Use Tools via Self-instruction**, NeurIPS 2023. [[Paper]](https://arxiv.org/abs/2305.18752)
  
  **ToolkenGPT: Augmenting Frozen Language Models with Massive Tools via Tool Embeddings**, NeurIPS 2023. [[Paper]](https://arxiv.org/abs/2305.11554)
  
  **Tool-Augmented Reward Modeling**, ICLR 2024. [[Paper]](https://arxiv.org/abs/2310.01045)
  
  **LLMs in the Imaginarium: Tool Learning through Simulated Trial and Error**, Preprint 2024. [[Paper]](https://arxiv.org/abs/2403.04746)
  
#### Response Generation.
- ##### Direct Insertion Methods
  
  **TALM: Tool Augmented Language Models**, Preprint 2022. [[Paper]](https://arxiv.org/abs/2205.12255)
  
  **Toolformer: Language Models Can Teach Themselves to Use Tools**, NeurIPS 2023. [[Paper]](https://arxiv.org/abs/2302.04761)
  
  **A Comprehensive Evaluation of Tool-Assisted Generation Strategies**, EMNLP 2023. [[Paper]](https://arxiv.org/abs/2310.10062)

- ##### Information Integration Methods
  
  **TPE: Towards Better Compositional Reasoning over Conceptual Tools with Multi-persona Collaboration**, Preprint 2023. [[Paper]](https://arxiv.org/abs/2309.16090)
  
  **RECOMP: Improving Retrieval-Augmented LMs with Compression and Selective Augmentation**, ICLR 2024. [[Paper]](http://arxiv.org/abs/2310.04408)
  
  **Learning to Use Tools via Cooperative and Interactive Agents**, Preprint 2024. [[Paper]](https://arxiv.org/abs/2403.03031)


### Benchmarks and Evaluation.
#### Benchmarks

| Benchmark | Reference | Description | #Tools | #Instances | Link | Release Time |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| API-Bank | [[Paper]](https://arxiv.org/abs/2304.08244) | Assessing the existing LLMs’ capabilities in planning, retrieving, and calling APIs. | 73 | 314 | [[Repo]](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/api-bank) | 2023-04 |
| APIBench | [[Paper]](http://arxiv.org/abs/2305.15334) | A comprehensive benchmark constructed from TorchHub, TensorHub, and HuggingFace API Model Cards. | 1,645 | 16,450 | [[Repo]](https://github.com/ShishirPatil/gorilla) | 2023-05 |
| ToolBench1  | [[Paper]](http://arxiv.org/abs/2305.16504) | A tool manipulation benchmark consisting of diverse software tools for real-world tasks. | 232 | 2,746 | [[Repo]](https://github.com/sambanova/toolbench) | 2023-05 |
| ToolAlpaca  | [[Paper]](http://arxiv.org/abs/2306.05301) | Evaluating the ability of LLMs to utilize previously unseen tools without specific training. | 426 | 3,938 | [[Repo]](https://github.com/tangqiaoyu/ToolAlpaca) | 2023-06 |
| RestBench  | [[Paper]](http://arxiv.org/abs/2306.06624) | A high-quality benchmark which consists of two real-world scenarios and human-annotated instructions with gold solution paths. | 94 | 157 | [[Repo]](https://restgpt.github.io) | 2023-06 |
| ToolBench2  | [[Paper]](http://arxiv.org/abs/2307.16789) | An instruction-tuning dataset for tool use, which is constructed automatically using ChatGPT. | 16,464 | 126,486 | [[Repo]](https://github.com/OpenBMB/ToolBench) | 2023-07 |
| MetaTool  | [[Paper]](http://arxiv.org/abs/2310.03128) | A benchmark designed to evaluate whether LLMs have tool usage awareness and can correctly choose tools. | 199 | 21,127 | [[Repo]](https://github.com/HowieHwong/MetaTool) | 2023-10 |
| TaskBench  | [[Paper]](https://arxiv.org/abs/2311.18760v2) | A benchmark designed to evaluate the capability of LLMs from different aspects, including task decomposition, tool invocation, and parameter prediction. | 103 | 28,271 | [[Repo]](https://github.com/microsoft/JARVIS) | 2023-11 |
| T-Eval  | [[Paper]](http://arxiv.org/abs/2312.14033) | Evaluating the tool-utilization capability step by step. | 15 | 533 | [[Repo]](https://github.com/open-compass/T-Eval) | 2023-12 |
| ToolEyes  | [[Paper]](http://arxiv.org/abs/2401.00741) | A fine-grained system tailored for the evaluation of the LLMs’ tool learning capabilities in authentic scenarios. | 568 | 382 | [[Repo]](https://github.com/Junjie-Ye/ToolEyes) | 2024-01 |
| UltraTool  | [[Paper]](http://arxiv.org/abs/2401.17167) | A novel benchmark designed to improve and evaluate LLMs’ ability in tool utilization within real-world scenarios. | 2,032 | 5,824 | [[Repo]](https://github.com/JoeYing1019/UltraTool) | 2024-01 |
| API-BLEND  | [[Paper]](http://arxiv.org/abs/2402.15491) | A large corpora for training and systematic testing of tool-augmented LLMs. | - | 189,040 | [[Repo]]() | 2024-02 |
| Seal-Tools  | [[Paper]](https://arxiv.org/abs/2405.08355) | Seal-Tools contains hard instances that call multiple tools to complete the job, among which some are nested tool callings. | 4,076 | 14,076 | [[Repo]](https://github.com/fairyshine/Seal-Tools) | 2024-05 |
| ToolQA  | [[Paper]](http://arxiv.org/abs/2306.13304) | It is designed to faithfully evaluate LLMs’ ability to use external tools for question answering.(QA) | 13 | 1,530 | [[Repo]](https://github.com/night-chen/ToolQA) | 2023-06 |
| ToolEmu  | [[Paper]](http://arxiv.org/abs/2309.15817) | A framework that uses a LM to emulate tool execution and enables scalable testing of LM agents against a diverse range of tools and scenarios.(Safety) | 311 | 144 | [[Repo]](https://github.com/ryoungj/toolemu) | 2023-09 |
| ToolTalk  | [[Paper]](http://arxiv.org/abs/2311.10775) | A benchmark consisting of complex user intents requiring multi-step tool usage specified through dialogue.(Conversation) | 28 | 78 | [[Repo]](https://github.com/microsoft/ToolTalk) | 2023-11 |
| VIoT  | [[Paper]](https://arxiv.org/abs/2312.00401) | A benchmark include a training dataset and established performance metrics for 11 representative vision models, categorized into three groups using semi-automated annotations.(VIoT) | 11 | 1,841 | [[Repo]]() | 2023-12 |
| RoTBench  | [[Paper]](http://arxiv.org/abs/2401.08326) | A multi-level benchmark for evaluating the robustness of LLMs in tool learning.(Robustness) | 568 | 105 | [[Repo]](https://github.com/Junjie-Ye/RoTBench) | 2024-01 |
| MLLM-Tool  | [[Paper]](http://arxiv.org/abs/2401.10727) | A system incorporating open-source LLMs and multimodal encoders so that the learnt LLMs can be conscious of multi-modal input instruction and then select the functionmatched tool correctly.(Multi-modal) | 932 | 11,642 | [[Repo]](https://github.com/MLLM-Tool/MLLM-Tool) | 2024-01 |
| ToolSword | [[Paper]](http://arxiv.org/abs/2402.10753) | A comprehensive framework dedicated to meticulously investigating safety issues linked to LLMs in tool learning.(Safety) | 100 | 440 | [[Repo]](https://github.com/Junjie-Ye/ToolSword) | 2024-02 |
| SciToolBench | [[Paper]](https://arxiv.org/abs/2402.11451) | Spanning five scientific domains to evaluate LLMs’ abilities with tool assistance.(Sci-Reasoning) | 2,446 | 856 | [[Repo]]() | 2024-02 |
| InjecAgent | [[Paper]](http://arxiv.org/abs/2403.02691) | A benchmark designed to assess the vulnerability of tool-integrated LLM agents to IPI attacks.(Safety) | 17 | 1,054 | [[Repo]](https://github.com/uiuc-kang-lab/InjecAgent) | 2024-02 |
| StableToolBench | [[Paper]](http://arxiv.org/abs/2403.07714) | A benchmark evolving from ToolBench, proposing a virtual API server and stable evaluation system.(Stable) | 16,464 | 126,486 | [[Repo]](https://github.com/zhichengg/StableToolBench) | 2024-03 |
| m&m's | [[Paper]](http://arxiv.org/abs/2403.11085) | A benchmark containing 4K+ multi-step multi-modal tasks involving 33 tools that include multi-modal models, public APIs, and image processing modules.(Multi-modal) | 33 | 4,427 | [[Repo]](https://github.com/RAIVNLab/mnms) | 2024-03 |
| ToolLens | [[Paper]](https://arxiv.org/abs/2405.16089) | ToolLens includes concise yet intentionally multifaceted queries that better mimic real-world user interactions. (Tool Retrieval) | 464 | 18,770 | [[Repo]]() | 2024-05 |

#### Evaluation
- Task Planning
  - Tool Usage Awareness
    
    **MetaTool Benchmark: Deciding Whether to Use Tools and Which to Use**, ICLR 2024. [[Paper]](http://arxiv.org/abs/2310.03128)
    
  - Pass Rate & Win Rate
    
    **ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs**, ICLR 2024. [[Paper]](https://arxiv.org/abs/2307.16789)
    
  - Accuracy
    
    **T-Eval: Evaluating the Tool Utilization Capability of Large Language Models Step by Step**, ICLR 2024. [[Paper]](http://arxiv.org/abs/2312.14033)
    
    **RestGPT: Connecting Large Language Models with Real-World RESTful APIs**, Preprint 2023. [[Paper]](https://arxiv.org/abs/2306.06624)
    
- Tool Selection
  
  - Recall
    
    **Recall, precision and average precision**,Department of Statistics and Actuarial Science 2004. [[Paper]](https://www.researchgate.net/publication/228874142_Recall_precision_and_average_precision)
  
  - NDCG
    
    **Cumulated gain-based evaluation of IR techniques**, TOIS 2002. [[Paper]](https://dl.acm.org/doi/10.1145/582415.582418)
  
  - COMP
    
    **COLT: Towards Completeness-Oriented Tool Retrieval for Large Language Models**, Preprint 2024. [[Paper]](https://arxiv.org/abs/2405.16089)
  
- Tool Calling
  - Consistent with stipulations
  
    **T-Eval: Evaluating the Tool Utilization Capability of Large Language Models Step by Step**, Preprint 2024. [[Paper]](http://arxiv.org/abs/2312.14033)
    
    **Planning and Editing What You Retrieve for Enhanced Tool Learning**, NAACL 2024. [[Paper]](https://arxiv.org/abs/2404.00450)
    
    **ToolEyes: Fine-Grained Evaluation for Tool Learning Capabilities of Large Language Models in Real-world Scenarios**, Preprint 2024. [[Paper3]](http://arxiv.org/abs/2401.00741)
    
- Response Generation
  - BLEU
    
    **Bleu: a Method for Automatic Evaluation of Machine Translation**, ACL 2002. [[Paper]](https://aclanthology.org/P02-1040/)
  - ROUGE
    
    **Rouge: A package for automatic evaluation of summaries**, ACL 2004. [[Paper]](https://aclanthology.org/W04-1013/)
  - Exact Match
    
    **cem: Coarsened exact matching in Stata**, The Stata Journal 2009. [[Paper]](https://gking.harvard.edu/files/cem-stata.pdf) 
  
### Challenges and Future Directions

- High Latency in Tool Learning
- Rigorous and Comprehensive Evaluation
- Comprehensive and Accessible Tools
- Safe and Robust Tool Learning
- Unified Tool Learning Framework
- Real-Word Benchmark for Tool Learning
- Tool Learning with Multi-Modal

### Other Resources
- #### Awesome Lists

  **ToolLearningPapers.** [[Repo]](https://github.com/thunlp/ToolLearningPapers)
  
  **awesome-tool-llm.** [[Repo]](https://github.com/zorazrw/awesome-tool-llm)
  
  **awesome-llm-tool-learning.** [[Repo]](https://github.com/AngxiaoYue/awesome-llm-tool-learning)

- #### Other Surveys
  
  **Augmented Language Models: a Survey**, TMLR 2024. [[Paper]](https://arxiv.org/abs/2302.07842)
  
  **Tool Learning with Foundation Models**, Preprint 2024. [[Paper]](http://arxiv.org/abs/2304.08354)
  
  **What Are Tools Anyway? A Survey from the Language Model Perspective**, Preprint 2024. [[Paper]](https://arxiv.org/abs/2403.15452)
  
