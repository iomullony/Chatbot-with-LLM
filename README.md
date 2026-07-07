Sánchez-O'Mullony Martínez, María Isabel, 123456

THD/DIT International Office Chatbot — RAG-based Q&A assistant using flan-t5-small

# Chatbot long description

## Task

The goal of this project was to build a Streamlit-based chatbot capable of answering questions about the Deggendorf Institute of Technology (DIT/THD), with a focus on the International Office. The target users are prospective and current international students asking about exchange programmes, application processes, language requirements, orientation, and contact information.

## Approach

The chatbot uses **Retrieval-Augmented Generation (RAG)** combined with a local Hugging Face language model (`google/flan-t5-small`). The approach was chosen over pure intent classification because it allows for flexible, open-ended questions without requiring a predefined set of intents.

The pipeline works as follows:

1. **Knowledge base**: A manually curated [`knowledge_base.txt`](knowledge_base.txt) file containing factual information about the DIT International Office, sourced from the official THD website. The file is split into paragraphs, each covering one topic (e.g. application deadlines, contact persons, language courses).

2. **Retrieval**: When the user asks a question, the `sentence-transformers` model (`all-MiniLM-L6-v2`) encodes both the question and all knowledge base chunks into vector embeddings. Cosine similarity is computed between the question embedding and all chunk embeddings to retrieve the top 3 most relevant chunks.

3. **Generation**: The retrieved chunks are injected into a structured prompt together with the user's question. The prompt instructs `flan-t5-small` to answer only based on the provided context. The model then generates a response using `AutoModelForSeq2SeqLM`.

4. **Logging**: Every user message and assistant response is automatically appended to `chatbot_history.log` using Python's `logging` module, with timestamps and session separators.

5. **UI**: The Streamlit interface follows a standard chat layout with message history. A sidebar toggle allows the user to inspect which knowledge base chunks were retrieved for each answer, which is useful for debugging.

The speech recognition and text-to-speech components from the original `code_02_chatbot.py` were removed, as microphone access was not available in the development environment (virtual machine).

# Run

Make sure all dependencies are installed:

```
pip install -r requirements.txt
```

Then start the Streamlit app:

```
streamlit run app.py
```

On first run, the models (`all-MiniLM-L6-v2` and `google/flan-t5-small`) will be downloaded automatically from Hugging Face (~400 MB total) and cached locally. Subsequent starts are faster.

# Used Hardware description

- **CPU**: Intel(R) Xeon(R) Silver 4214 @ 2.20 GHz, 4 virtual cores (1 thread per core)
- **RAM**: 15 GB total, ~12 GB available during development
- **GPU**: None — all inference runs on CPU

Running entirely on CPU means model loading takes approximately 30–60 seconds on first start. Response generation typically takes 5–15 seconds per query depending on prompt length. Despite the lack of GPU, the small model sizes of `flan-t5-small` and `all-MiniLM-L6-v2` make the system usable in practice.

# Discussion of the results

The following dialogs were taken from `chatbot_history.log` and illustrate both strengths and weaknesses of the chatbot.

---
DIALOG 1: Incoming Exchange Student — Application Deadlines & Process
Target: International Office (Incomings & Deadlines)
---

It answers correctly the application period and the question about sending physical documents by mail. However, the response about whether the application form can be filled out right now is incorrect.

```
USER:  Hello! I want to come to DIT as an exchange student for the winter semester. When is the application period?
ASSISTANT: 1 April to 1 June for October (winter semester) entries, and 1 October to 1 December for March (summer semester) entries
USER:  Can I just go ahead and fill out the application form on your website right now?
ASSISTANT: Yes
USER:  but it is july
ASSISTANT: Orientation Week
USER:  do I need to mail my physical documents to Deggendorf after completing the form?
ASSISTANT: no
```

**Analysis**: The date retrieval and the answer about physical documents are correct — the knowledge base clearly states that the application portal is only active during the application window and that everything is done online via Mobility Online. However, when the user provides contextual follow-up information ("but it is july"), the model fails completely. `flan-t5-small` has no reasoning capability over the conversation context — it treats each prompt independently and cannot combine the retrieved date information with the user's follow-up. This is a fundamental limitation of the model size and architecture.

---
DIALOG 3: Language Centre & TestDaF Fees
Target: International Office / Language Centre
---

The German language requirement and the TestDaF exam question are answered correctly. However, the contact for exam registration is wrong.

```
USER:  Hello! Are exchange students required to learn German while studying at DIT?
ASSISTANT: All exchange students are required to participate in German language courses during their semester. DIT offers free German language courses at levels A1 to C1. An intensive German course at A1 level is also offered during Orientation Week. Students can obtain official German language certificates at DIT, including TestDaF and telc exams. DIT is a licensed examination centre for TestDaF. The TestDaF exam fee is €210. To register, students should contact the Language Centre at sprachenzentrum@th-deg.de.
USER:  Can I take an official language exam like the TestDaF at the university?
ASSISTANT: Yes
USER:  Whom should I contact if I want to register for the exam?
ASSISTANT: incomings@th-deg.de
```

**Analysis**: The first answer is a strong positive example — the model correctly retrieved and reproduced the full relevant chunk from the knowledge base. However, in the third turn the model gave the wrong email address. The correct contact for TestDaF registration is `sprachenzentrum@th-deg.de`, which was even mentioned in the previous answer. This shows that the model does not maintain or reference its own prior outputs within the same conversation, and when multiple email addresses are present in the retrieved chunks, it may select the wrong one.

---
DIALOG 5: Specific Contact Persons for Programmes
Target: International Office (Staff & Contacts)
---

The coordinator name is correct but the phone number is wrong. The General Business contact is correct.

```
USER:  Hello, I have a detailed question about the General Engineering exchange programme structure. Who is the coordinator?
ASSISTANT: Nadine Kappl
USER:  What is her telephone number?
ASSISTANT: +49 991 3615-0
USER:  What is her telephone number of Nadine Kappl if I need to call her?
ASSISTANT: +49 991 3615-0
USER:  And who is the contact person for the General Business programme?
ASSISTANT: Johanna Maurer
```

**Analysis**: The model correctly identified Nadine Kappl and Johanna Maurer from the knowledge base. However, it consistently returned the general university switchboard number (+49 991 3615-0) instead of Nadine Kappl's direct number (0991/3615-242). Both numbers appear in the knowledge base, and the model retrieved the wrong one — likely because the general number appears earlier in the retrieved chunks and the model defaults to the first plausible number it finds.

# Description of technical problems during the development process

- **`text2text-generation` pipeline error**: The installed version of `transformers` did not support the `text2text-generation` task string in the `pipeline()` constructor. This was resolved by loading `flan-t5-small` directly using `AutoTokenizer` and `AutoModelForSeq2SeqLM` and calling `model.generate()` manually.

- **Streamlit `torchvision` warnings**: On startup, Streamlit's file watcher tried to inspect all submodules of the `transformers` library, including vision models that require `torchvision`. Since `torchvision` was not installed, this produced a stream of `ModuleNotFoundError` warnings. These did not affect functionality and were suppressed with `--logger.level=error`.

- **Microphone unavailable in VM**: The speech recognition and TTS features from the original `code_02_chatbot.py` could not be tested or used, as the virtual machine environment does not expose a microphone device. These components were removed from the final implementation.

# Suggestions for improvements

- **Larger or more capable model**: `flan-t5-small` is very limited in its ability to reason across multiple turns and disambiguate between similar facts (e.g. multiple phone numbers or email addresses). A larger model such as `flan-t5-base` or `SmolLM-135M-Instruct` would likely produce more accurate and context-aware answers.

- **Conversation history in the prompt**: Currently, each prompt only contains the retrieved chunks and the current question. Including the last 2–3 turns of conversation history in the prompt would allow the model to handle follow-up questions and contextual references (e.g. "but it is july") much more reliably.

- **Larger and more structured knowledge base**: The knowledge base could be expanded with more specific information, such as individual course descriptions, partner university lists, and FAQs sourced directly from the official THD website. Shorter, more focused chunks would also improve retrieval precision.

- **Re-ranking retrieved chunks**: Instead of using only cosine similarity for retrieval, a cross-encoder re-ranker could be added to re-score the top-k candidates and select the most relevant chunk more accurately — reducing cases where the wrong phone number or email is retrieved.

- **Adding speech input/output**: Once a microphone is available, the original voice recognition and TTS components from `code_02_chatbot.py` could be re-integrated, making the chatbot more accessible and interactive.