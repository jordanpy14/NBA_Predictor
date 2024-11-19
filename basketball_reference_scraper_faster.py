import os
import asyncio
from tqdm import tqdm
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout
import requests
import pandas as pd
from io import StringIO
import random

# Constants
# focus on 1985 and onward py 
SEASONS = list(range(2006, 2015))
DATA_DIR = "data"
STANDINGS_DIR = os.path.join(DATA_DIR, "standings")
SCORES_DIR = os.path.join(DATA_DIR, "scores")

proxies = [
    {"ip": "190.119.80.140", "port": "80"},
    {"ip": "43.201.121.81", "port": "80"},
    {"ip": "116.203.139.209", "port": "5153"},
    {"ip": "195.26.247.26", "port": "6969"},
    {"ip": "72.10.164.178", "port": "1417"},
    {"ip": "129.21.108.112", "port": "8080"},
    {"ip": "63.35.64.177", "port": "3128"},
    {"ip": "204.236.176.61", "port": "3128"},
    {"ip": "64.147.212.78", "port": "8080"},
    {"ip": "160.86.242.23", "port": "8080"},
    {"ip": "54.67.125.45", "port": "3128"},
    {"ip": "222.252.194.29", "port": "8080"},
    {"ip": "13.208.56.180", "port": "80"},
    {"ip": "204.236.137.68", "port": "80"},
    {"ip": "46.51.249.135", "port": "3128"},
    {"ip": "3.37.125.76", "port": "3128"},
    {"ip": "54.152.3.36", "port": "80"},
    {"ip": "43.202.154.212", "port": "80"},
    {"ip": "43.200.77.128", "port": "3128"},
    {"ip": "18.228.198.164", "port": "80"},
    {"ip": "44.219.175.186", "port": "80"},
    {"ip": "52.67.10.183", "port": "80"}
]

# Ensure directories exist
os.makedirs(STANDINGS_DIR, exist_ok=True)
os.makedirs(SCORES_DIR, exist_ok=True)


def parse_html(box_score):
    with open(box_score) as f:
        html = f.read()

    soup = BeautifulSoup(html, 'html.parser')
    [s.decompose() for s in soup.select("tr.over_header")]
    [s.decompose() for s in soup.select("tr.thead")]
    return soup

def read_season_info(soup):
    nav = soup.select("#bottom_nav_container")[0]
    hrefs = [a["href"] for a in nav.find_all('a')]
    season = os.path.basename(hrefs[1]).split("_")[0]
    return season

def get_teams_from_bottom_nav(soup:BeautifulSoup):
    # Locate the bottom navigation container
    bottom_nav = soup.find("div", id="bottom_nav_container")
    
    # Find all list items (teams) within the navigation container
    team_links = bottom_nav.find_all("a", href=True)
    
    # Extract team names and their links if they contain "/teams/" in the href
    teams = []
    for link in team_links:
        if "/teams/" in link['href']:
            team_name = link.text.replace(" Schedule", "").strip()
            team_url = link['href'].split("/")[2]
            teams.append(team_url)
            # teams.append({"team_name": team_name, "team_url": team_url})
    
    return teams

def read_line_score(soup):
    line_score = pd.read_html(StringIO(str(soup)), attrs = {'id': 'line_score'})[0]
    cols = list(line_score.columns)
    cols[0] = "team"
    cols[-1] = "total"
    line_score.columns = cols
    
    line_score = line_score[["team", "total"]]
    
    return line_score

def read_stats(soup, team, stat):
    df = pd.read_html(StringIO(str(soup)), attrs = {'id': f'box-{team}-game-{stat}'}, index_col=0)[0]
    # Use BeautifulSoup to parse each player row and extract the player identifier
    players = soup.select(f'#box-{team}-game-{stat} tbody tr')
    player_ids = []

    for player in players:
        player_id = player.find('th').get('data-append-csv')
        player_ids.append(player_id)

    # Check if there's an extra row in df (like "Team Totals") and adjust player_ids accordingly
    if len(player_ids) < len(df):
        player_ids.append("Team Totals")  # Add a placeholder for the last row

    # Add the player_ids as a new column in the DataFrame
    df['player_id'] = player_ids
    # df = df.apply(pd.to_numeric, errors="coerce")
    return df

def read_stats_quarter(soup, team, quarter, stat):
    df = pd.read_html(StringIO(str(soup)), attrs = {'id': f'box-{team}-{quarter}-{stat}'}, index_col=0)[0]
    # Use BeautifulSoup to parse each player row and extract the player identifier
    # players = soup.select(f'#box-{team}-{quarter}-{stat} tbody tr')
    # player_ids = []

    # for player in players:
    #     player_id = player.find('th').get('data-append-csv')
    #     player_ids.append(player_id)

    # # Check if there's an extra row in df (like "Team Totals") and adjust player_ids accordingly
    # if len(player_ids) < len(df):
    #     player_ids.append("Team Totals")  # Add a placeholder for the last row

    # # Add the player_ids as a new column in the DataFrame
    # df['player_id'] = player_ids
    # df = df.apply(pd.to_numeric, errors="coerce")
    return df

def get_team_records(soup):
    # Extract both team divs from the scorebox
    team_divs = soup.find_all('div', class_='scores', limit=2)  # Assuming there are always two teams listed as in your HTML structure

    results = []

    for team_div in team_divs:
        team_name = team_div.find_previous('strong').a.text
        record_div = team_div.find_next_sibling('div')
        record_text = record_div.text.strip()
        results.append((team_name, record_text))

    # # Print the results
    # for result in results:
    #     print(f"Team: {result[0]}, Record: {result[1]}")
        
    return results

async def get_html(browser, url, selector, sleep=5, retries=3):
    ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36 Edg/116.0.1938.81"
    context = await browser.new_context(user_agent=ua)
    page = await context.new_page()
    html = None
    html = None
    for i in range(1, retries + 1):
        # print(f"Attempt {i} on {url}")
        try:
            await asyncio.sleep(sleep * i)
            await page.goto(url, timeout=30000)  # 60s timeout
            await page.wait_for_selector(selector, timeout=60000)
            html = await page.inner_html(selector)
            break
        except PlaywrightTimeout:
            print(f"Timeout error on {url} attempt {i}")
            continue
        finally:
            await page.close()
    return html

async def get_html_requester(browser, url, selector, sleep=5, retries=3):
    proxy = random.choice(proxies)
    proxy_server = f"http://{proxy['ip']}:{proxy['port']}"

    # Create a new browser context with the randomly selected proxy
    context = await browser.new_context(ignore_https_errors=True, proxy={"server": proxy_server})
    page = await context.new_page()
    # page = await browser.new_page()
    html = None
    for i in range(1, retries + 1):
        # print(f"Attempt {i} on {url}")
        try:
            await asyncio.sleep(sleep * i)
            await page.goto(url, timeout=3000)  
            await page.wait_for_selector(selector, timeout=6000)
            html = await page.inner_html(selector)
            break
            # await page.goto(url, timeout=30000)  # 60s timeout
            # await page.wait_for_selector(selector, timeout=60000)
            # html = await page.inner_html(selector)
            # response = requests.get(url)
            # soup = BeautifulSoup(response.text, 'html.parser')
            # Wait for the selector and get HTML of the selected element
            # Note: BeautifulSoup does not support waiting, it processes what's in the response
            # element = soup.select_one(selector)   
            # if element:
            #     html = element.decode_contents() 
            # break
        except PlaywrightTimeout:
            print(f"Timeout error on {url} attempt {i}")
            continue
        finally:
            await page.close()
            
    if not html:
        print(f"Failed to get HTML from {url}")
        
    return html

def format_minutes(mp_str):
    mp_str = str(mp_str)  # Ensure input is treated as a string
    if ':' in mp_str:  # Check if the format seems correct
        minutes, seconds = map(int, mp_str.split(':'))
        return minutes + seconds / 60.0
    else:
        return 0.0  # Default value for non-played games or invalid entries


def flatten_player_stats(sorted_players, quarter, name):
    """ Flatten the top 5 players' stats into a single row with custom column names. """
    top_players = sorted_players.head(5)
    flattened_stats = {}
       # Define explicit mappings for column names
    column_mapping = {
        '_3': 'FGPer',
        '_4': '3P',
        '_5': '3PA',
        '_6': '3PPer',
        '_9': 'FTPer',
        '_20': 'plus_minus'
    }
    
    for i, player_stats in enumerate(top_players.itertuples(index=False), start=1):
        for stat in player_stats._fields:
            # Replace special characters in stat names to prevent truncation or errors
            if stat in column_mapping:
                safe_stat_name = column_mapping[stat]
            else:
                safe_stat_name = stat
            flattened_stats[f"{quarter}_player{i}_{name}_{safe_stat_name}"] = getattr(player_stats, stat)
    
    return pd.DataFrame([flattened_stats])

async def scrape_season(browser, season):
    url = f"https://www.basketball-reference.com/leagues/NBA_{season}_games.html"
    html = await get_html(browser, url, "#content .filter") 
    soup = BeautifulSoup(html, 'html.parser')
    links = soup.find_all("a")
    # links for all months in the season
    standings_pages = [f"https://www.basketball-reference.com{l['href']}" for l in links if 'href' in l.attrs]
    # print("standing pages", standings_pages)

    # Iterating over all months in a season 
    for i, url in enumerate(tqdm(standings_pages, desc=f"Scraping {season}")):
        # if i < 4:
        #     continue
        # print("Focused on", url)
        first, second = os.path.basename(url).split("-")
        save_path = os.path.join(STANDINGS_DIR, f"{first}_{i}-{second}")
        # if os.path.exists(save_path):
        #     continue
        
        # for all games in the season per month
        html_origin = await get_html(browser, url, "#all_schedule")  
        soup = BeautifulSoup(html_origin, 'html.parser')
        season_date = first + "_" + str(i) + "-" + second
        links = soup.find_all("a")
        box_scores = [f"https://www.basketball-reference.com{l.get('href')}" for l in links if l.get('href') and "boxscores" in l.get('href') and "html" in l.get('href')]
        # print("box_scores", box_scores)

        season_date = season_date.split(".")[0]
        games = []
        # Scraping games for one month 
        for url_box in tqdm(box_scores, desc=f"Scraping {season_date}"):
            # print("Focused on", url_box)

            # save_path_game = os.path.join(SCORES_DIR, f"{season_date}-{os.path.basename(url)}")

            # saving game hmtl
            soup = await get_html(browser, url_box, "#content")  
            # if soup:
            #     with open(save_path_game, "w+") as f:
            #         f.write(soup)
            soup = BeautifulSoup(soup, 'html.parser')
            #save soup to file
   
            
            [s.decompose() for s in soup.select("tr.over_header")]
            [s.decompose() for s in soup.select("tr.thead")]
            # if html:
            #     with open(save_path, "w+") as f:
            #         f.write(html)

            gameid = os.path.basename(url_box).split("/")[-1].split(".")[0]
            # soup = parse_html(html)

            # line_score = read_line_score(soup)
            # teams = list(line_score["team"])
            teams = get_teams_from_bottom_nav(soup)
            team_record = get_team_records(soup)
            # print("team_record:", "home", team_record[0], "away", team_record[1])
            summaries = []
            game_teams = pd.DataFrame({'gameid': [gameid], 'Home': [teams[0]], 'Away': [teams[1]],  'season': [read_season_info(soup)], 'date': [pd.to_datetime(os.path.basename(url_box)[:8], format="%Y%m%d")]})
            summaries.append(game_teams)
            for index, team in enumerate(teams):               
                for quarter in ["q1", "q2", "h1"]:
                    basic = read_stats_quarter(soup, team, quarter, "basic")
                    team_basic = basic.iloc[-1, :].copy()
                    name = 'home' if index == 0 else 'away'
                    team_basic = team_basic.rename(lambda x: f"{quarter}_team_{name}_{x}".replace('%', 'Per').replace('+/-', 'plus_minus'))
                    team_basic_transposed = team_basic.to_frame().transpose() 
                    team_basic_transposed.reset_index(drop=True, inplace=True) 
                    # print("team_basic transposed shape", team_basic_transposed.shape)
                    summaries.append(team_basic_transposed)
                    
                    players_basic = basic.iloc[:-1, :].copy()
                    # Convert MP to float for sorting
                    players_basic['MP_float'] = players_basic['MP'].apply(format_minutes)
                    # Sort by the new MP_float column and flatten the top 5 player stats
                    sorted_players = players_basic.sort_values(by='MP_float', ascending=False)
                    flattened_row = flatten_player_stats(sorted_players, quarter, name)
                    flattened_row.reset_index(drop=True, inplace=True)
                    # print("flattened_row shape", flattened_row.shape)
                    summaries.append(flattened_row)
                    
                basic = read_stats_quarter(soup, team, 'game', "basic")
                team_basic = basic.iloc[-1, :].copy()
                name = 'home' if index == 0 else 'away'
                team_basic = team_basic.rename(lambda x: f"game_team_{name}_{x}".replace('%', 'Per').replace('+/-', 'plus_minus'))
                team_basic_transposed = team_basic.to_frame().transpose() 
                team_basic_transposed.reset_index(drop=True, inplace=True) 
                # print("team_basic transposed shape", team_basic_transposed.shape)
                summaries.append(team_basic_transposed)      

                advanced = read_stats_quarter(soup, team, 'game', "advanced")
                team_advanced = advanced.iloc[-1, :].copy()
                name = 'home' if index == 0 else 'away'
                team_advanced = team_advanced.rename(lambda x: f"game_team_{name}_{x}".replace('%', 'Per').replace('+/-', 'plus_minus'))
                team_advanced_transposed = team_advanced.to_frame().transpose() 
                team_advanced_transposed.reset_index(drop=True, inplace=True) 
                # print("team_basic transposed shape", team_advanced_transposed.shape)
                summaries.append(team_advanced_transposed)      

            
            team_record = pd.DataFrame({'team_home_record': [team_record[0][1]], 'team_away_record': [team_record[1][1]]})
            summaries.append(team_record)
    
            summary = pd.concat(summaries, axis=1)
            # summary.to_csv(f"game_{gameid}.csv", index=False)   
            games.append(summary)

        all_games = pd.concat(games, ignore_index=True)
        os.makedirs(f"data/{season}", exist_ok=True)
        all_games.to_csv(f"data/{season}/{season_date}_all_games_summary.csv", index=False)
        print(f"Data saved to data/{season}/{season_date}_all_games_summary.csv")
            


async def main():
    async with async_playwright() as p:
        # browser = await p.chromium.launch()
        browser = await p.firefox.launch(headless=True)
        try:
            for season in tqdm(SEASONS, desc="All seasons"):
                await scrape_season(browser, season)
        finally:
            await browser.close()  



if __name__ == "__main__":
    asyncio.run(main())
    # asyncio.run(main2())
    # main3()