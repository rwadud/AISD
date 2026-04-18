# Artificial Intelligence Software Development

**CST8510 Week 11**
Dr. Hari M Koduvely

---

## Agenda for Today

- **Theory: 5:30PM – 7:30PM**
  - Prompt Optimization
  - Reflection Methods
  - Advanced RAG Techniques
- **Lab: 7:30PM – 9:30PM**
  - Standup Meetings

---

## Prompt Optimization

### Uncertainty Thresholding with Self-Consistency

#### "Opt-out" confidence thresholding

- **Implicit approach:** LLM self-confidence evaluation assigns human support, general LLM, or a user-friendly "I don't know" message as a prompt response "out"
- **Explicit approach:** Rule-based threshold through self-consistency
  - **Uncertainty:** Proportion of answers that are unique
  - **Token likelihood:** Likelihood of an LLM output (cumulative token probability)

**Example system prompt (virtual air travel agent):**

> SYSTEM
>
> I want you to act as a virtual air travel agent with expertise in customer support that receives, as input, customer requests for information and assistance pertaining to the end-to-end airport experience. Please categorize each request into one of the following categories:
>
> - Flight Bookings: Requests for flight bookings, cancellations, modifications, or upgrades
> - Flight Information: Requests for flight details, live schedule information, boarding times, safety protocol, and other flight procedures
> - Flight Resources: Requests to track or identify baggage, locate transportation services post-flight, or find available curbside and parking locations and resources pre-flight
>
> For example, a user request of "I'm noticing some heavy rain in my area. Should I expect any delays for my flight that departs in 2 hours?" would receive a Flight Information label.
>
> Please only provide the request label in your response. If you are unsure of the proper label to assign based on the context above, please label the request with Human Support so the customer can get directed to a live agent.

**Prompt components annotated:**
- Persona introduction
- Capability articulation
- Definition of scope
- Placeholder for multi-shot
- Output specification with "opt-out"

*Source: Beyond Fine-tuning Approaches in LLM Optimization by Superwise*

---

## Prompt Optimization

### Uncertainty Thresholding with Self-Consistency

#### Optimizing prompt accuracy: Self-consistency

Improve the consistency of an existing LLM system without altering its architecture.

**Chain-of-thought prompting (single path):**

- Prompt → LLM → "If you need to land by 1 PM, you should consider the following: 1:00 PM (desired arrival time) - 1 hour (flight time) = 12:00 PM. ... So: 12:00 PM - 30 minutes = 11:30 AM"
- **The answer is 11:30AM**

**Self-consistency (multiple reasoning paths):**

User prompt: *"I'm flying from BNA to ATL. If I need to land by 1PM local time, by what time should my flight leave?"*

System prompt:
> You are a calculator. Take the computational process **step by step**.
>
> Q: I have a flight from MSP to MIA departing at 9AM local time. What's my expected arrival time?
>
> A: Start with the expected flight time: 3 hours and 30 minutes. Add the flight time to the departure time: 9AM + 3 hours + 30 minutes = 12:30PM. Account for any time zone changes: 12:30PM + 1 hour = 1:30PM. ...

Sample a diverse set of reasoning paths:

1. "The flight from Nashville (BNA) to Atlanta (ATL) typically takes around 1 hour [...] you would need to depart from Nashville no later than **11 AM CST** to account for the 1-hour flight time."
2. "To answer this question, we need to consider two main factors: the flight duration and the time difference between the two locations [...] you should aim to take off from Nashville at least 1 hour before, so around **12PM local time**."
3. "Flight Duration: The average flight duration from BNA to ATL is approximately 1 hour [...] aim for a flight that leaves around **11AM local time** from BNA to ensure you land in ATL by 1PM."

Marginalize out reasoning paths to aggregate final answers → **The answer is 11:00AM**

*Source: Beyond Fine-tuning Approaches in LLM Optimization by Superwise*

---

## Prompt Optimization

### Uncertainty Thresholding with Self-Consistency

**Pipeline:**

Chain-of-Thought Prompt ("Step-by-Step") → Diversified Reasoning Paths (Self-Consistency) → Uncertainty Threshold Application → Reasoning Path Selection

**Sample outputs from self-consistency:**
- [...] 11 AM CST [...]
- [...] 12PM local time [...]
- [...] 11AM local time [...]
- [...] 12:30 PM CST [...]
- [...] 11AM CST [...]

**Decision logic:**
- If **Unique Output Proportion < 0.5** → *The answer is 11:00AM*
- Else → *"I don't know."* → route to one of:
  - General LLM
  - Human Support
  - User-Friendly Message

*Source: Beyond Fine-tuning Approaches in LLM Optimization by Superwise*

---

## Prompt Optimization

### Uncertainty Thresholding with Self-Consistency

#### Active prompting with uncertainty thresholding

Iterative loop:

1. Sample of prospective new shots (Automated or human-engineered)
2. Diverse answers (CoT + self-consistency)
3. Uncertainty threshold application
4. Top-N most uncertain shots
5. Add to shot set (LLM context)
6. Shot set evaluation (Token constraints, bias-variance, completeness, etc.)
7. Loop back to step 1

*Source: Beyond Fine-tuning Approaches in LLM Optimization by Superwise*

---

## OPRO – Optimization by PROmpting

- OPRO is a simple yet effective technique that employs LLMs as optimizers.
- The optimization task is described in natural language.
- LLM generates new solutions based on the problem description and previously found solutions.
- This iterative process converges to an optimal solution.

*Reference: Large Language Models as Optimizers*

---

## OPRO – How it Works?

- **Meta-Prompt:** This is the input to the LLM, containing the problem description, solution constraints, and previously evaluated solutions with their corresponding scores.
- **LLM as Optimizer:** The LLM generates new candidate solutions based on the information in the meta-prompt.
- **Objective Function Evaluator:** This component evaluates the new solutions and provides feedback to the LLM, which is then used to refine the meta-prompt for the next optimization step.

*Reference: Large Language Models as Optimizers*

---

## OPRO – Illustration

**GSM8K Dataset – Grade School Math Problems Dataset**

| Prompt | Score |
|---|---|
| Take a deep breath and work on this problem step-by-step. | **80.2** |
| Break this down. | 79.9 |
| A little bit of arithmetic and a logical approach will help us quickly arrive at the solution to this problem. | 78.5 |
| Let's combine our numerical command and clear thinking to quickly and accurately decipher the answer. | 74.5 |

*Reference: Large Language Models as Optimizers*

---

## Self-reflection

### Self-evaluation: Reflexion

- For use cases with a ground truth that is out of reach: a feedback loop that iteratively approaches high-confidence outputs
  a. Correction of errors, especially for multi-step reasoning or code generation tasks
  b. Improvement of output quality, especially for requirement satisfaction
  c. Boost in contextual understanding through evaluation of interpretations
- Two core approaches:
  a. **Feedback-Driven Reflexion:** Iterative changes via self-assessment are triggered by negative user feedback (thumbs up/down feature for chat-style interfaces, etc.)
  b. **Implicit Reflexion:** Self-assessment is applied on all requests before initial front-end delivery

**Pipeline:** Task → Proposed Solution → Internal Tests → Self-reflection → Refined Solution (with Test Generation feeding Internal Tests)

*Source: Beyond Fine-tuning Approaches in LLM Optimization by Superwise*
*Reference: https://promptengineering.org/reflexion-an-iterative-approach-to-llm-problem-solving/*

---

## Self-reflection

### Feedback-driven reflexion

**User:** *My flight has been delayed. Where can I go for more information on my flight's status?*

**Initial response (labeled poor):** *I'm sorry to hear that! For more details on your flight, please visit our Flight Status page.*

**Feedback-Driven Self-Reflection Layer:**

- If response was labeled poor (General): *Please modify the **relevance** of your previous response to the customer's request for up-to-date flight status information. Offer a more **specific and helpful** solution.*
- If response was labeled incomplete (Specific): *Please modify the [**completeness**, accuracy, helpfulness, specificity] of your previous response to the customer's request for up-to-date flight status information. Offer a more [**complete**, accurate, helpful, specific] solutions.*

**Refined response:**
> I'm sorry to hear that! For more information on your delayed flight, please consider consulting one of the following resources:
> 1. Airport Website: Please visit our Flight Status page for real-time information about flight status, delays, and any changes to the schedule.
> 2. Airline Kiosk: Ask an airport representative for updated flight details before entering a security checkpoint
> 3. Airport Information Screens: Your airport will display real-time flight information on departure and arrival screens located throughout the terminals.

*Source: Beyond Fine-tuning Approaches in LLM Optimization by Superwise*

---

## Self-reflection

### Implicit reflexion

**User:** *My flight has been delayed. Where can I go for more information on my flight's status?*

**Initial response:** *I'm sorry to hear that! For more details on your flight, please visit our Flight Status page.*

**Hidden Self-Reflection Layer:**
> Evaluate the **relevance** of your previous response to the customer's issue of intermittent internet connection. Offer a more specific and helpful solution **if necessary**.

**Refined response:**
> I'm sorry to hear that! For more information on your delayed flight, please consider consulting one of the following resources:
> 1. Airport Website: Please visit our Flight Status page for real-time information about flight status, delays, and any changes to the schedule.
> 2. Airline Kiosk: Ask an airport representative for updated flight details before entering a security checkpoint
> 3. Airport Information Screens: Your airport will display real-time flight information on departure and arrival screens located throughout the terminals.

*Source: Beyond Fine-tuning Approaches in LLM Optimization by Superwise*

---

## Advanced RAG

### Naive RAG

**Pipeline:**

query → Embedding model → Vector store Index → Database → context → LLM → answer

(query also flows directly to the LLM alongside the retrieved context)

*Reference: Advanced RAG Techniques: an Illustrated Overview*

---

## Advanced RAG

### Advanced RAG

**Pipeline components:**

- **Agents:** Query transformation → Query routing (produces transformed query, list of queries, and tool choice)
- **Fusion retrieval** → DB storage → **Reranking postprocessing** → retrieved context → LLM → answer
- **Vector store Index** and **Summary index** both feed into DB storage
- query also flows directly to LLM

*Reference: Advanced RAG Techniques: an Illustrated Overview*

---

## Advanced RAG – Improvement in Chunking

### Unstructured chunking

- **Character splitting (size-based):** Chunking text into equally sized documents based on character length, which may result in:
  - Multiple topics in one chunk
  - One topic in multiple chunks
- **Unstructured chunking:** Content-aware identification of text objects via syntactical rules
- **Benefit:** Assigning one chunk to each text object produces semantically consistent documents of varying lengths
- **Drawback:** Syntactical similarity + proximity =/= semantic similarity in every case

*Reference: https://medium.com/unstructured-io/rag-isnt-so-easy-why-llm-apps-are-challenging-and-how-unstructured-can-help-8daaf859c615*
*Source: Advanced RAG Techniques: an Illustrated Overview*

---

## Advanced RAG – Improvement in Chunking

### Semantic chunking (Contextual compression)

- Leverages sentence-level semantic similarity methods in order to order chunk-level content more meaningfully
  - Split documents into sentences
  - Sentence embeddings
  - Cluster semantically similar sentences and re-indexing
- Built-in **contextual compression** minimizes unneeded noise sent to the LLM
- **Benefit:** Semantically tight chunks ensure minimal loss
- **Drawback:** Contextual richness of the original corpus structure is lost

*Reference: https://medium.com/@hadiazouni/text-splitting-chunking-for-rag-applications-7ccbb6dcc9f9*
*Source: Advanced RAG Techniques: an Illustrated Overview*

---

## Advanced RAG – Extending Context Window

### Sentence Window Retrieval

Given a query like *"Why A23a is moving?"*, the retrieval returns the matching sentence along with surrounding context (the window), which is then passed to the LLM.

**Example passage:**

> The largest iceberg, A23a, is a massive ice shelf that calved from the Antarctic coastline in 1986 and was grounded in the Weddell Sea for over 30 years. It spans about 1,500 square miles, making it more than twice the size of Greater London and about three times the size of New York City. It is approximately 400 meters (1,312 feet) thick, making it a true colossus of ice.
>
> **Recently, A23a has broken free from the ocean floor and is now drifting in the open sea, heading towards the South Atlantic on a path known as "iceberg alley."**
>
> If it reaches South Georgia, it could disrupt the foraging routes of seals, penguins, and other seabirds, preventing them from feeding their young properly. There are also concerns that it could cause disruptions to shipping if it heads toward South Africa, potentially leading to collisions and other hazards for maritime traffic. A23a's movement is being closely monitored, as it could have significant impacts on the environment and human activities.

The **extended context** (full passage) is sent to the LLM rather than just the matching sentence.

*Reference: Advanced RAG Techniques: an Illustrated Overview*

---

## Advanced RAG – Hybrid Search

### Fusion retrieval / hybrid search

**Pipeline:**

- query + Documents → Vector index → Top k results
- query + Documents → sparse n-grams index (BM25) → Top k results
- Both top-k result sets → Reciprocal Rank Fusion → Top n → LLM → answer

*Reference: Advanced RAG Techniques: an Illustrated Overview*

---

## Advanced RAG – Reranking

### RAG Pipeline with Re-ranking

**Flow:** Query Input + Document Collection → Initial Retrieval (Bi-Encoders) → **Re-ranking (Cross-Encoders)** → Content Generation → Response

- Re-ranking is the highlighted component
- Initial Retrieval is a standard RAG component

*Reference: Relevance Revolution — how Re-ranking Transforms Rag Systems*

---

## Advanced RAG – Reranking

**Relevance Scoring Models:**
- The system assigns relevance scores to each document
- Use models like BERT-based rerankers or cross-encoders
- Considers both the query and retrieved result together

**Traditional Ranking Algorithms:**
- BM25 (considers term frequency and document length)

*Reference: Relevance Revolution — how Re-ranking Transforms Rag Systems*

---

## Advanced RAG – Reranking

- Can improve quality of retrieval significantly
- Prone to increasing the latency
- Additional computational cost also can incur

*Reference: Relevance Revolution — how Re-ranking Transforms Rag Systems*

---

## Advanced RAG – Fine-Tuning Embedding Models

- Embedding models are trained using standard dataset corpus
- They may not capture the deep semantic relationships in a document
- Fine-tune embedding models with your domain data helps
- More cost effective than fine-tuning LLM

---

## Summary of Today's Learning

- Different prompt optimization methods.
- Reflection methods – self and feedback driven.
- Improving RAG through:
  - Better Chunking
  - Reranking
  - Fine-tuning embedding models

---

## Thank You
