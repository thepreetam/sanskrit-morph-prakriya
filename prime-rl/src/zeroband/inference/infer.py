import time

if hasattr(config, 'task_type') and config.task_type == "sanskrit_morph":
    # For Sanskrit, directly use the prepared prompts (metadata strings)
    prompts = [m[0]["content"] for m in messages] # Messages for sanskrit are just [{'role':'user', 'content': metadata_str}]
elif tokenizer.chat_template:
    prompts = tokenizer.apply_chat_template(messages, tokenize=False, continue_final_message=True)
    if config.model_name != "Qwen/QwQ-32B":
        for i, p in enumerate(prompts):
            prompts[i] = p.replace("<｜begin of sentence｜>", "") # Corrected line
else:
    prompts = fake_chat_template(messages)

start_time = time.time() 