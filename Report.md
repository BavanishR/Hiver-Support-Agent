What Good support means for this project:
    -Tone: Customer tweets should be replied to using empathic tone that follows the platform's character limit while not making any unwarranted financial promises.

    -Customer Intent Triage: The reliable categorization of customers' messages across 7 core customer intents to determine if automated routing is appropriate or human intervention is required.

    -Next Steps: Giving customers clear steps on what to do next (e.g. asking the customer to send a direct message with order number and email) when it comes to intents such as DELIVERY_DELAY, ORDER_STATUS, and DIGITAL_CONTENT_APP.

    -Safe Escalation Route: Automatically recognizing high-friction intents (REFUND_AND_RETURN, WRONG_DAMAGED_ITEM, ACCOUNT_ACCESS) and sending them straight to human queues instead of risking self-resolution.

What I doesn't want to build:
    -There is no automatic resolution of requests for ACCOUNT_ACCESS and password reset in public forums because of the extreme privacy and security issues involved; the requests are routed to human security workflows right away. 

    -It is optimized for short-form interactions via social media platforms  and does not allow email, live-chat stream, or telephone conversation transcription.

    -The agent doesn’t process any refund transactions, change any shipping information, or order replacements in live Amazon databases. The system is strictly limited to intent classification, RAG retrieval, and queue routing functions.

Baselines asked as per Assignment:

    In order to gauge the effectiveness of our proposed LangGraph agent for customer support, we used two baselines for comparison. The first is a trivial baseline, where the prediction is done on the basis of the majority class, UNCLEAR_VAGUE. This gives us a lower-bound of 20.5% accuracy in terms of intent classification. A simple baseline, which consists of a zero-shot LLM classifier (without RAG and confidence guardrails), has 52.0% accuracy. Our proposed multi-node architecture has an accuracy of 70.0% intent classification and 72.0% escalation classification, which is 18.0% better.

| Architecture Tier       | Intent Accuracy | Escalation Accuracy | System Description                                      |
|-------------------------|-----------------|---------------------|---------------------------------------------------------|
| Trivial Baseline        | 20.5%           | 0.0%                | Predicts majority class (UNCLEAR_VAGUE)                 |
| Simple Baseline         | 52.0%           | ~48.0%              | Single zero-shot LLM prompt without RAG or state guardrails |
| Proposed LangGraph Agent| 70.0%           | 72.0%              | Multi-node graph + ChromaDB RAG + 0.70 Confidence Routing | 

Failure Modes:

    - Before i introduced clarification loop most of the time the intents were classified into UNCLEAR_VAGUE so i thought of introducing a clarification loop which is limited to just 2 times because if we not get clarified we will get stuck inside a infinite loop

    - While classifying the intents some queries like "The Delivery date is 2 days ag0 i haven't received it yet so i need refund" these cases seem to like multi-intent but if we go with delivery delay we cant make it ascalate to a human so I handled the delivery delay cases to DM them

    - For some sarcasm wordings like  "Great job Amazon, 3 days late again, your prime fast shipping is really working out!"Actuallly fell into UNCLEAR_VAGUE but the actual is DELIVERY_DELAY a zero-shot reply turned into a clerification loop with a low accuracy

    - The are many multilinggual queries raised from the customer side although they are easy to handle our embedding model doesn't contain multilanguage support i make the START -> Clarrify language node so if any other language  inputs are given we can easily escalate it.

    - Some escalation queries were raised to auto-handling due not improper customer inputs to solve this some intents were handled with replying to the user and then escalating it .


What is Misleading About my System's Headline Numbers:

    Point1:

        Illusion: A mean LLM Judge score of 3.78 out of 5.0 creates the illusion of a very dependable model.

        Truth: The LLM Judge checks politeness, formatting, and tone of the answer. The LLM judge often scores empathic, friendly replies a 4 out of 5 even though the node is classified incorrectly (some node interprets a refund inquiry as a delivery complaint).

    Point2:

        Illusion: It is taken for granted that automated LLM evaluation corresponds with human evaluation.
        
        Reality: The evaluation I performed found only a low correlation between human ground-truth routing and the LLM judge score. Dependence on LLM judge solely as a performance indicator gives rise to illusions of safety.

    Point3:

        Illusion: The 70% Intent Accuracy is a good score to deploy in production.

        Reality: The 70% Intent Accuracy was computed on a set of only 50 samples in order to meet the deadline. The real world has more noise such as slang language, sarcasm, non-English languages, and multi-issues customer tweets.

    Point4:

        Although my model manages to deliver a 72% rate of escalation success and 3.78/5 mean response quality scores, these are numbers that paint an inaccurate picture of the production readiness of our system. Our evaluation tool showed that there is low Pearson correlation ($r = 0.123$) between LLM-as-a-Judge scores and routing correctness based on ground truth. The LLM judge rates mostly response tone, empathy, and formatting, giving many responses a 4/5 rating despite them being wrong in their assessment of the customer’s intent.

Things I will do with one more week:

    - I will develop a local model to classify intents without relying on a LLM By which we can reduce token costs and the time to Classify the input into a intent

    - I will develop a interactive GUI frontend which can run in the Browser and I will Design Backend Using FastAPI and make it into a real application instead of just completing given task

    - I can also do the application remembers customer session state even when the server is restarted and during asynchronous  Backend  processing cycles by providing them ID's. 

    - I was actually planned do a advanced retrieval mechanism Combine dense vector embeddings with sparse keyword search (BM25) and some other advanced hybrid search mechanisms.

    - I also wish to add Semantic Caching to this Project but due to time constraints i couldn't complete within 1 week i will complete it.

    - I will add a LLM as a Judge mechanism to re write the response if it is not good at all

    - I will also like to  some other features like a application in which they can choose their own intents and they can easily get rid of their problems in turn

    - Since you asked you should get rrun the application within 15 mins with the readme i haven't ingested that much data into the chroma DB so for industry ready work i will ingest all those data to make the application respond better