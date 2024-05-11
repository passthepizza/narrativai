import discord
from discord.ext import commands
import asyncio
import os
import openai
import aiohttp
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import re
import logging
import json
import subprocess
import math

logging.basicConfig(level=logging.ERROR)

load_dotenv()

openai.api_key = os.environ["OPENAI_API_KEY"]
openai.api_base = "https://api.deepinfra.com/v1/openai"

intents = discord.Intents.default()
intents.typing = False
intents.presences = False
intents.message_content = True
intents.members = True
client = commands.Bot(command_prefix='!', intents=intents)

async def create_rp_bot(user, story_type, story, bot_token, response_style, bot_creator_id):

    test_client = discord.Client(intents=discord.Intents.default())
    await test_client.login(bot_token)
    bot_id = test_client.user.id
    await test_client.close()

    bot_folder = f"rp_bot_{bot_id}"
    os.makedirs(bot_folder, exist_ok=True)

    story = story.replace("\n", " ")

    if response_style == "short":
        rp_prompt = "..."
        max_tokens = 300
    else:
        rp_prompt = "..."
        max_tokens = 500

    config = {
        "BOT_TOKEN": bot_token,
        "CHARACTER_DESCRIPTION": story.strip(),
        "RESPONSE_STYLE": response_style,
        "RP_PROMPT": rp_prompt,
        "MAX_TOKENS": max_tokens,
        "BOT_CREATOR_ID": user.id  
    }
    with open(f"{bot_folder}/config.json", "w") as f:
        json.dump(config, f)

    subprocess.Popen(["nohup", "python3", "bot.py", str(bot_id)])

async def extract_text_from_webpage(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            html = await response.text()
            soup = BeautifulSoup(html, "html.parser")
            text = soup.get_text()
            try:
                text = await summarize_text(text)
            except openai.error.InvalidRequestError as e:
                logging.error(f"OpenAI API Error: {str(e)}")
                raise Exception("Unable to summarize the text. Please try again later.")
            return text

async def summarize_text(text):
    prompt = f"Please summarize the following text in detail:\n\n{text}"
    try:
        response = await openai.ChatCompletion.acreate(
            model="microsoft/WizardLM-2-8x22B",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes text. You don't use any markup or quotations marks. Focus on capturing the character and don't include any meta info."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,
            n=1,
            stop=None,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except openai.error.InvalidRequestError as e:
        logging.error(f"OpenAI API Error: {str(e)}")
        raise

async def is_valid_bot_token(token):
    try:
        test_client = discord.Client(intents=discord.Intents.default())
        await test_client.login(token)
        await test_client.close()
        return True
    except discord.LoginFailure:
        return False

async def get_bot_user(bot_token):
    try:
        bot = discord.Client(intents=discord.Intents.default())
        await bot.login(bot_token)
        app_info = await bot.application_info()
        await bot.close()
        return app_info
    except Exception as e:
        logging.error(f"Error retrieving bot user: {str(e)}")
        return None

@client.event
async def on_ready():
    print(f"Logged in as {client.user.name} (ID: {client.user.id})")

class StoryTypeView(discord.ui.View):
    def __init__(self, user):
        super().__init__(timeout=60.0)
        self.user = user
        self.story_type = None
        self.canceled = False

    @discord.ui.button(label="Custom Description", custom_id="custom_story", style=discord.ButtonStyle.primary)
    async def custom_story_button_callback(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.story_type = "custom_story"
        button.style = discord.ButtonStyle.success
        button.emoji = "✅"
        button.disabled = True
        await interaction.response.edit_message(view=self)
        self.stop()

    @discord.ui.button(label="Link", custom_id="link", style=discord.ButtonStyle.primary)
    async def link_button_callback(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.story_type = "link"
        button.style = discord.ButtonStyle.success
        button.emoji = "✅"
        button.disabled = True
        await interaction.response.edit_message(view=self)
        self.stop()

    @discord.ui.button(label="Cancel", custom_id="cancel", style=discord.ButtonStyle.danger)
    async def cancel_button_callback(self, interaction: discord.Interaction, button: discord.ui.Button):
        button.style = discord.ButtonStyle.danger
        button.emoji = "❌"
        button.disabled = True
        await interaction.response.edit_message(view=self)
        self.canceled = True
        self.stop()

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        return interaction.user == self.user

class ResponseStyleView(discord.ui.View):
    def __init__(self, user):
        super().__init__(timeout=60.0)
        self.user = user
        self.response_style = None
        self.canceled = False

    @discord.ui.button(label="Short Responses (Like c.ai)", custom_id="short_responses", style=discord.ButtonStyle.primary)
    async def short_responses_button_callback(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.response_style = "short"
        button.style = discord.ButtonStyle.success
        button.emoji = "✅"
        button.disabled = True
        await interaction.response.edit_message(view=self)
        self.stop()

    @discord.ui.button(label="Detailed Responses (Novel RP)", custom_id="detailed_responses", style=discord.ButtonStyle.primary)
    async def detailed_responses_button_callback(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.response_style = "detailed"
        button.style = discord.ButtonStyle.success
        button.emoji = "✅"
        button.disabled = True
        await interaction.response.edit_message(view=self)
        self.stop()

    @discord.ui.button(label="Cancel", custom_id="cancel", style=discord.ButtonStyle.danger)
    async def cancel_button_callback(self, interaction: discord.Interaction, button: discord.ui.Button):
        button.style = discord.ButtonStyle.danger
        button.emoji = "❌"
        button.disabled = True
        await interaction.response.edit_message(view=self)
        self.canceled = True
        self.stop()

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        return interaction.user == self.user

@client.event
async def on_message(message):

    if message.author == client.user:
        return

    if isinstance(message.channel, discord.DMChannel) and client.user.mentioned_in(message):

        story_type_view = StoryTypeView(message.author)

        embed = discord.Embed(title="Create an RP Bot", description="Select how you want to create the bot:", color=discord.Color.blue())
        onboarding_message = await message.channel.send(embed=embed, view=story_type_view)

        await story_type_view.wait()

        if story_type_view.canceled:
            return

        story = ""  

        if story_type_view.story_type == "custom_story":
            embed = discord.Embed(title="Character info", description="Please provide character info:", color=discord.Color.blue())
            await message.channel.send(embed=embed)
            story_message = await client.wait_for("message", check=lambda m: m.author == message.author and isinstance(m.channel, discord.DMChannel))
            story = story_message.content
        elif story_type_view.story_type == "link":
            embed = discord.Embed(title="Link", description="Please provide the URL of your character:", color=discord.Color.blue())
            await message.channel.send(embed=embed)
            url = await client.wait_for("message", check=lambda m: m.author == message.author and isinstance(m.channel, discord.DMChannel))
            url = url.content
            try:
                story = await extract_text_from_webpage(url)
            except Exception as e:
                logging.error(f"Text Extraction Error: {str(e)}")
                embed = discord.Embed(title="Error", description=str(e), color=discord.Color.red())
                await message.channel.send(embed=embed)
                return

        response_style_view = ResponseStyleView(message.author)
        embed = discord.Embed(title="Response Style", description="Select the desired response style for the bot:", color=discord.Color.blue())
        await message.channel.send(embed=embed, view=response_style_view)

        await response_style_view.wait()

        if response_style_view.canceled:
            return

        response_style = response_style_view.response_style

        embed = discord.Embed(title="Bot Configuration", description="Please provide the following information:", color=discord.Color.blue())
        embed.add_field(name="Intents", value="Make sure to turn on the following intents:\n- Presence Intent\n- Server Members Intent\n- Message Content Intent", inline=False)
        embed.add_field(name="Bot Token", value="Please provide a valid bot token:", inline=False)
        await message.channel.send(embed=embed)

        while True:
            bot_token = await client.wait_for("message", check=lambda m: m.author == message.author and isinstance(m.channel, discord.DMChannel))
            bot_token = bot_token.content

            if await is_valid_bot_token(bot_token):
                break
            else:
                embed = discord.Embed(title="Invalid Bot Token", description="The provided bot token is invalid. Please provide a valid bot token.", color=discord.Color.red())
                await message.channel.send(embed=embed)

        await create_rp_bot(message.author, story_type_view.story_type, story, bot_token, response_style, message.author.id)

        embed = discord.Embed(title="RP Bot Created", description="The RP bot has been created successfully!", color=discord.Color.green())
        await message.channel.send(embed=embed)

def list_rp_bots():
    bot_folders = [folder for folder in os.listdir(".") if folder.startswith("rp_bot_")]
    bot_info = []
    for bot_folder in bot_folders:
        with open(f"{bot_folder}/config.json", "r") as f:
            config = json.load(f)
        bot_info.append({
            "Bot ID": bot_folder.split("_")[2],
            "Creator ID": config["BOT_CREATOR_ID"],
            "Character Description": config["CHARACTER_DESCRIPTION"][:50] + "..."  
        })
    return bot_info

def stop_rp_bot(bot_id):
    bot_folder = f"rp_bot_{bot_id}"
    if os.path.exists(bot_folder):
        subprocess.run(["pkill", "-f", f"python3 bot.py {bot_id}"])
        return f"RP bot {bot_id} stopped successfully."
    else:
        return f"RP bot {bot_id} not found."

def restart_rp_bot(bot_id):
    bot_folder = f"rp_bot_{bot_id}"
    if os.path.exists(bot_folder):
        subprocess.run(["pkill", "-f", f"python3 bot.py {bot_id}"])
        subprocess.Popen(["nohup", "python3", "bot.py", str(bot_id)])
        return f"RP bot {bot_id} restarted successfully."
    else:
        return f"RP bot {bot_id} not found."

def restart_all_rp_bots():
    bot_folders = [folder for folder in os.listdir(".") if folder.startswith("rp_bot_")]
    restarted_bots = []
    for bot_folder in bot_folders:
        bot_id = bot_folder.split("_")[2]
        subprocess.run(["pkill", "-f", f"python3 bot.py {bot_id}"])
        subprocess.Popen(["nohup", "python3", "bot.py", str(bot_id)])
        restarted_bots.append(bot_id)
    return f"Restarted bots: {', '.join(restarted_bots)}"

def create_bot_list_embed(bot_info, current_page, total_pages):
    embed = discord.Embed(title="Running RP Bots", color=discord.Color.blue())
    start_index = (current_page - 1) * 10
    end_index = min(start_index + 10, len(bot_info))

    for i in range(start_index, end_index):
        bot = bot_info[i]
        embed.add_field(name=f"Bot ID: {bot['Bot ID']}", value=f"Creator ID: {bot['Creator ID']}\nCharacter Description: {bot['Character Description']}", inline=False)

    embed.set_footer(text=f"Page {current_page}/{total_pages}")
    return embed

def is_authorized_user(user: discord.User):
    return user.id == 710473094749487134

class BotListView(discord.ui.View):
    def __init__(self, bot_info, current_page, total_pages):
        super().__init__(timeout=None)
        self.bot_info = bot_info
        self.current_page = current_page
        self.total_pages = total_pages

    @discord.ui.button(label="Previous", style=discord.ButtonStyle.blurple)
    async def previous_button_callback(self, interaction: discord.Interaction, button: discord.ui.Button):
        if self.current_page > 1:
            self.current_page -= 1
            embed = create_bot_list_embed(self.bot_info, self.current_page, self.total_pages)
            await interaction.response.edit_message(embed=embed, view=self)

    @discord.ui.button(label="Next", style=discord.ButtonStyle.blurple)
    async def next_button_callback(self, interaction: discord.Interaction, button: discord.ui.Button):
        if self.current_page < self.total_pages:
            self.current_page += 1
            embed = create_bot_list_embed(self.bot_info, self.current_page, self.total_pages)
            await interaction.response.edit_message(embed=embed, view=self)

@client.tree.command(name="list_bots", description="List all running RP bots")
async def list_bots_command(interaction: discord.Interaction):
    if not is_authorized_user(interaction.user):
        await interaction.response.send_message("You are not authorized to use this command.")
        return

    bot_info = list_rp_bots()
    if bot_info:
        current_page = 1
        total_pages = math.ceil(len(bot_info) / 10)
        embed = create_bot_list_embed(bot_info, current_page, total_pages)
        view = BotListView(bot_info, current_page, total_pages)
        await interaction.response.send_message(embed=embed, view=view)
    else:
        await interaction.response.send_message("No running RP bots found.")

@client.tree.command(name="stop_bot", description="Stop an RP bot")
async def stop_bot_command(interaction: discord.Interaction, bot_id: str):
    if not is_authorized_user(interaction.user):
        await interaction.response.send_message("You are not authorized to use this command.")
        return

    result = stop_rp_bot(bot_id)
    await interaction.response.send_message(result)

@client.tree.command(name="restart_bot", description="Restart an RP bot")
async def restart_bot_command(interaction: discord.Interaction, bot_id: str):
    if not is_authorized_user(interaction.user):
        await interaction.response.send_message("You are not authorized to use this command.")
        return

    result = restart_rp_bot(bot_id)
    await interaction.response.send_message(result)

@client.tree.command(name="restart_all_bots", description="Restart all RP bots")
async def restart_all_bots_command(interaction: discord.Interaction):
    if not is_authorized_user(interaction.user):
        await interaction.response.send_message("You are not authorized to use this command.")
        return

    result = restart_all_rp_bots()
    await interaction.response.send_message(result)

@client.tree.command(name="bot_info", description="Get information about a specific RP bot")
async def bot_info_command(interaction: discord.Interaction, bot_id: str):
    if not is_authorized_user(interaction.user):
        await interaction.response.send_message("You are not authorized to use this command.")
        return

    bot_folder = f"rp_bot_{bot_id}"
    if os.path.exists(bot_folder):
        with open(f"{bot_folder}/config.json", "r") as f:
            config = json.load(f)

        bot_token = config["BOT_TOKEN"]
        app_info = await get_bot_user(bot_token)
        if app_info is None:
            await interaction.response.send_message(f"Failed to retrieve information for RP bot {bot_id}.")
            return

        character_description = config["CHARACTER_DESCRIPTION"]
        if len(character_description) > 1024:
            character_description = character_description[:1021] + "..."

        embed = discord.Embed(title=f"RP Bot {bot_id} Information", color=discord.Color.blue())
        embed.set_author(name=app_info.name, icon_url=app_info.icon.url if app_info.icon else None)
        embed.add_field(name="Creator ID", value=config["BOT_CREATOR_ID"], inline=False)
        embed.add_field(name="Character Description", value=character_description, inline=False)
        embed.add_field(name="Response Style", value=config["RESPONSE_STYLE"], inline=False)

        await interaction.response.send_message(embed=embed)
    else:
        await interaction.response.send_message(f"RP bot {bot_id} not found.")

@client.event
async def on_ready():
    print(f"Logged in as {client.user.name} (ID: {client.user.id})")
    await client.tree.sync()  

client.run(os.environ["DISCORD_BOT_TOKEN"])
