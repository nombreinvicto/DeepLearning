import random, numpy as np

words_list = ['apple bat', 'babt cat', 'hello world']
tries_allowed = 6
attempts = 0

computer_choice = random.choice(words_list)
computer_choice_array = np.array(list(computer_choice))
word_length = len(computer_choice)
print(f"Word chosen by computer: {computer_choice}")
initial_user_guess = [" "] * word_length
print(initial_user_guess)
init_nill_pos = np.where(np.array(initial_user_guess) == " ")[0]

while True:
    user_input = input(f"Please guess a character: ")
    user_input = user_input.lower()
    # make sure input is a single char
    if len(user_input) == 1:
        # now check if char is in word chosen by pc
        if user_input in computer_choice_array:
            print("found the char!, Choose the next char!")

            # found out where the chars are present
            char_pos = np.where(computer_choice_array == user_input)[0]
            for pos in char_pos:
                initial_user_guess[pos] = user_input
            print(initial_user_guess)
            init_nill_pos = np.where(np.array(initial_user_guess) == " ")[0]
            print("=" * 50)

            # check if all chars have been gueesed correctly
            nill_pos = np.where(np.array(initial_user_guess) == " ")[0]
            if len(nill_pos) == 1 and nill_pos == init_nill_pos:
                print("You have guessed all chars right!")
                print("=" * 50)
                break

        else:
            print("Try again! Chars dont match!")
            attempts += 1
            print(f"{tries_allowed - attempts} tries left")
            print(initial_user_guess)
            print("=" * 50)

            if attempts >= tries_allowed:
                print("You ran out of all tries! Play again")
                print("=" * 50)
                break

    else:
        # this happens in case of invalid input
        print("Please enter input of length 1")
        print("=" * 50)
        continue
