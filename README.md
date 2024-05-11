# NarrativAI
This project is about creating Discord roleplaying bots that can interact with users in a detailed or concise manner based on the configured response style. These bots can be created via a character description given by the user OR summarised via scraping a given link


## Requirements
- Python 3.8+
- Discord.py library
- aiohttp
- BeautifulSoup4
- Python-dotenv
- OpenAI
- Annoy (for embedding generation and similarity)
- Subprocess
- OS
- Logging
- JSON
- Math

Ensure that all dependencies are installed using the following command:
```shell
pip install discord.py aiohttp beautifulsoup4 python-dotenv openai==0.28 annoy
```

---

## Environment Variables
Create a `.env` file in the root directory of your project, including the following variables:
```
DISCORD_BOT_TOKEN=your_discord_bot_token
OPENAI_API_KEY=your_openai_api_key
DEEPINFRA_API_KEY=your_deepinfra_api_key
```

---

## Usage

### Running the Bot
To run the main bot (`main.py`), use:
```shell
python main.py
```

For each RP bot that is created, a separate `bot.py` instance is spawned. It reads its configuration from a generated JSON file and uses environment variables for API keys.

### Bot Creation
Ping the hosted bot in DMs to start the bot creation process <3
1. **Story Type:** Choose between using a custom description for your character or providing a link to a webpage that details the character.
2. **Response Style:** Select the desired response style (`Short Responses` or `Detailed Responses`).
3. **Bot Token:** Enter a valid Discord bot token. Ensure your bot has the required intents enabled.

### Advanced Features
- **Add World Info:** Custom command to add themed information to the bot's memory for more contextual responses.
- **Clear History:** Command to clear the conversation history and reset context.
- **Set Persona:** Tailor how the bot addresses or references you in the channel by setting a persona.

---

## Note
Ensure you have the necessary permissions and intents enabled on your Discord bot at https://discord.com/developers. The required intents are:
- Guild Messages
- Guilds
- Direct Messages
- Message Content Intent

---


## Contributions
Contributions are welcome! Please create a pull request or issue if you have suggestions.
---
