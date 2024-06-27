#A template on how to condense the user prompt for better querying
# custom_prompt = PromptTemplate(
#     """\
# Given a conversation (between Human and Assistant) and a follow up message from Human, \
# rewrite the message to be a standalone question that captures all relevant context \
# from the conversation.

# <Chat History>
# {chat_history}

# <Follow Up Message>
# {question}

# <Standalone question>
# """
# )

# chat_engine = CondenseQuestionChatEngine.from_defaults(
#     query_engine=query_engine,
#     condense_question_prompt=custom_prompt,
#     verbose=True,
# )

# while (prompt := input("Enter a prompt(q to quit):")) != "q":
#     result=chat_engine.stream_chat(prompt)
#     print(result)
