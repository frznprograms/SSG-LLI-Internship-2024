def GenerateNewData(system_prompt, user_prompt_1, user_prompt_2, num_datapoints, API_KEY):
    client = OpenAI(api_key = API_KEY)

    # First API call to prepare the model
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{user_prompt_1}"}
        ]
    )

    reply = completion.choices[0].message.content
    if reply is not None and "ready" in reply.lower():
        final_case_list = []

        # Generate new case notes
        for _ in range(num_datapoints):  # Assume each API call generates 25 notes
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt_2}
                ]
            )
            answer = completion.choices[0].message.content
            case_list = answer.split('A')
            final_case_list.extend(case_list)

        # Trim the final case list if it exceeds num_datapoints
        #final_case_list = final_case_list[:num_datapoints]

        final_label_list = []
        for case_note in final_case_list:
            labeling_prompt = f"Based on the examples you have studied, classify the following case note into one of the categories: {case_note}\n\n\
            Reply with the category number 0, 1, 2, or 3 only and no other text."
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": labeling_prompt}
                ]
            )
            answer = completion.choices[0].message.content
            try:
                final_label = int(answer.strip())
            except ValueError:
                final_label = random.randint(0, 4)
            final_label_list.append(final_label)

    return final_case_list, final_label_list
