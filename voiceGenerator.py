import os
import json
import yaml
from utils import CustomObject, get_yaml_loader
import time
from graphAPI.graphs import getContent
from Generator.utils import content_stats
from Generator.OpenAI import convert as openAIConvert
from Generator.Maya import convert as mayaConvert
from dotenv import load_dotenv

load_dotenv(override=True)

if "GRAPH_ACCESS_TOKEN" not in os.environ:
    raise Exception("Load Graph Access token first !")

access_token = os.getenv("GRAPH_ACCESS_TOKEN")

with open('cache.json') as f:
    CACHE = json.load(f)

with open('contentCache.json') as f:
    CONTENT_CACHE = json.load(f)


def updateCache():
    with open('contentCache.json', 'w') as f:
        json.dump(CONTENT_CACHE, f)


def main():

    with open('config.yaml', 'r') as file:
        config = yaml.load(file, get_yaml_loader())

    x = json.dumps(config)
    Args = json.loads(x, object_hook=lambda d: CustomObject(**d))

    update_cache = False

    notebook_name = Args.Graph.NotebookName
    section_name = Args.Graph.SectionName

    print(f"Running voice generation for {notebook_name} {section_name}")

    pages = CACHE["Pages"][section_name]

    pages = pages if not Args.Generator.Pages or Args.Generator.Pages == "all" else pages[:Args.Generator.Pages]

    for page in pages:
        if CONTENT_CACHE and page["title"] in CONTENT_CACHE:
            content = CONTENT_CACHE[page["title"]]
        else:
            content = getContent(page["id"])
            CONTENT_CACHE[page["title"]] = content
            update_cache = True

        if update_cache: updateCache()

        print(f"Downloaded the content {page['title']} : {content_stats(content)}")

        if Args.Generator.OpenAI.Action:
            openAIConvert(Args, content, page["title"])
        elif Args.Generator.Maya.Action:
            mayaConvert(Args, content, page["title"])

        time.sleep(30)


if __name__ == '__main__':
    main()