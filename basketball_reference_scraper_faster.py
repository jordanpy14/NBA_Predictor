import os
import asyncio
from tqdm import tqdm
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout
import requests
import pandas as pd
from io import StringIO

# Constants
# focus on 1985 and onward 
SEASONS = list(range(1988, 2023))
DATA_DIR = "data"
STANDINGS_DIR = os.path.join(DATA_DIR, "standings")
SCORES_DIR = os.path.join(DATA_DIR, "scores")


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

async def get_html(browser, url, selector, sleep=5, retries=3):
    page = await browser.new_page()
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
    page = await browser.new_page()
    html = None
    for i in range(1, retries + 1):
        # print(f"Attempt {i} on {url}")
        try:
            await asyncio.sleep(sleep * i)
            # await page.goto(url, timeout=30000)  # 60s timeout
            # await page.wait_for_selector(selector, timeout=60000)
            # html = await page.inner_html(selector)
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            # Wait for the selector and get HTML of the selected element
            # Note: BeautifulSoup does not support waiting, it processes what's in the response
            element = soup.select_one(selector)   
            if element:
                html = element.decode_contents() 
            break
        except PlaywrightTimeout:
            print(f"Timeout error on {url} attempt {i}")
            continue
        finally:
            await page.close()
    return html

async def scrape_season(browser, season):
    url = f"https://www.basketball-reference.com/leagues/NBA_{season}_games.html"
    html = await get_html_requester(browser, url, "#content .filter") 
    soup = BeautifulSoup(html, 'html.parser')
    links = soup.find_all("a")
    # links for all months in the season
    standings_pages = [f"https://www.basketball-reference.com{l['href']}" for l in links if 'href' in l.attrs]
    # print("standing pages", standings_pages)

    # Iterating over all months in a season 
    for i, url in enumerate(tqdm(standings_pages, desc=f"Scraping {season}")):
        # print("Focused on", url)
        first, second = os.path.basename(url).split("-")
        save_path = os.path.join(STANDINGS_DIR, f"{first}_{i}-{second}")
        # if os.path.exists(save_path):
        #     continue
        
        # for all games in the season per month
        html_origin = await get_html_requester(browser, url, "#all_schedule")  
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

            # save_path = os.path.join(SCORES_DIR, f"{season_date}-{os.path.basename(url)}")

            soup = await get_html_requester(browser, url_box, "#content")  
            soup = BeautifulSoup(soup, 'html.parser')
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
            summaries = []
            for team in teams:
                basic = read_stats(soup, team, "basic")
                # print("basic", basic)
                advanced = read_stats(soup, team, "advanced")
                # print("advanced", advanced)

                # totals = pd.concat([basic.iloc[-1,:], advanced.iloc[-1,:]])
                # totals.index = totals.index.str.lower()

                # maxes = pd.concat([basic.iloc[:-1].max(), advanced.iloc[:-1].max()])
                # maxes.index = maxes.index.str.lower() + "_max"

                summary = pd.concat([basic.iloc[:-1], advanced.iloc[:-1]], axis=1)

                # Remove duplicate columns
                summary = summary.loc[:, ~summary.columns.duplicated()]
                # print("summary", summary)
                
                summaries.append(summary)
            summary = pd.concat(summaries)

            summary["gameid"] = gameid
            summary["season"] = read_season_info(soup)
            summary["date"] = os.path.basename(url_box)[:8]
            summary["date"] = pd.to_datetime(summary["date"], format="%Y%m%d")
            summary["unique_id"] = summary["gameid"] + "_" + summary["player_id"]
            games.append(summary)

        all_games = pd.concat(games, ignore_index=True)
        # Reorder columns to have 'unique_id' as the first column
        columns_order = ["unique_id"] + [col for col in all_games.columns if col != "unique_id"]
        all_games = all_games[columns_order]
        os.makedirs(f"data/{season}", exist_ok=True)
        all_games.to_csv(f"data/{season}/{season_date}_all_games_summary.csv", index=False)
        print(f"Data saved to data/{season}/{season_date}_all_games_summary.csv")
            


async def main():
    async with async_playwright() as p:
        # browser = await p.chromium.launch()
        browser = await p.firefox.launch()
        try:
            for season in tqdm(SEASONS, desc="All seasons"):
                await scrape_season(browser, season)
        finally:
            await browser.close()  
 






if __name__ == "__main__":
    asyncio.run(main())
    # asyncio.run(main2())
    # main3()