import os
import json
import yaml
import time
import argparse
from Emotions.sanitise import sanitise
from Emotions.emotions import addEmotions
from utils import CustomObject, get_yaml_loader
from GraphAPI.graphs import GraphAPI
from Generator.utils import content_stats
from Generator.OpenAI import convert as openAIConvert
from Generator.Maya import convert as mayaConvert


class VoiceGenerator:

    def __init__(self):

        self.CONTENT_CACHE = {}
        self.VOICE_CACHE = {}
        self.EMOTION_CACHE = {}

        parser = argparse.ArgumentParser(description="Initidate data")
        parser.add_argument("--config", type=str, default="Default", help="Configuration file")
        args = parser.parse_args()

        with open('default.yaml', 'r') as file:
            config = yaml.load(file, get_yaml_loader())

        x = json.dumps(config)
        self.Args = json.loads(x, object_hook=lambda d: CustomObject(**d))

        self.Args.Platform = args.config

        with open('cache.json') as f:
            self.CACHE = json.load(f)

        self.createCache('contentCache.json')
        self.createCache('voiceCache.json')
        self.createCache('emotionCache.json')

    def createCache(self, file):

        if os.path.isfile(file):
            with open(file) as f:
                if file == 'contentCache.json':
                    self.CONTENT_CACHE = json.load(f)
                elif file == 'voiceCache.json':
                    self.VOICE_CACHE = json.load(f)
                elif file == 'emotionCache.json':
                    self.EMOTION_CACHE = json.load(f)
        else:
            with open(file, 'w') as f:
                if "content" in file:
                    json.dump(self.CONTENT_CACHE, f)
                elif "voice" in file:
                    json.dump(self.VOICE_CACHE, f)
                elif "emotion" in file:
                    json.dump(self.EMOTION_CACHE, f)

    def updateCache(self):

        for file in ['contentCache.json', 'voiceCache.json', 'emotionCache.json']:
            with open(file, 'w') as f:
                if file == 'contentCache.json':
                    json.dump(self.CONTENT_CACHE, f, indent=2, ensure_ascii=False)
                elif file == 'voiceCache.json':
                    json.dump(self.VOICE_CACHE, f, indent=2, ensure_ascii=False)
                elif file == 'emotionCache.json':
                    json.dump(self.EMOTION_CACHE, f, indent=2, ensure_ascii=False)

    def generation(self):

        update_cache = False

        notebook_name = self.Args.Graph.NotebookName
        section_name = self.Args.Graph.SectionName

        print(f"Running voice generation for {notebook_name} {section_name}")

        pages = self.CACHE["Pages"][section_name]

        if self.Args.Generator.Pages:
            pages = pages[:self.Args.Generator.Pages]

        graph = GraphAPI(self.Args.Graph)

        for pageNo, page in enumerate(pages):
            if not self.Args.Graph.RefreshPages and self.CONTENT_CACHE and page["title"] in self.CONTENT_CACHE:
                content = self.CONTENT_CACHE[page["title"]]
            else:
                content = graph.getContent(page["id"])
                self.CONTENT_CACHE[page["title"]] = content
                update_cache = True

            if update_cache: self.updateCache()

            # print(content)

            print(f"Processing the content {page['title']} : {content_stats(content)}")

            if self.VOICE_CACHE and page["title"] in self.VOICE_CACHE:
                spell_checked_lines = self.VOICE_CACHE[page["title"]]
            else:
                spell_checked_lines = sanitise(self.Args, content)
                self.VOICE_CACHE[page["title"]] = spell_checked_lines
                update_cache = True

            if update_cache: self.updateCache()

            # print(f"Creating Emotions for {page['title']}")

            # if self.EMOTION_CACHE and page["title"] in self.EMOTION_CACHE:
            #     emotion_lines = self.EMOTION_CACHE[page["title"]]
            # else:
            #     emotion_lines = addEmotions(self.Args, content)
            #     self.EMOTION_CACHE[page["title"]] = emotion_lines
            #     update_cache = True
            #
            # if update_cache: self.updateCache()

            # print(f"Generating voice for {page['title']}")

            # spell_checked_lines

            # if self.Args.Generator.OpenAI.Action:
            #     openAIConvert(self.Args, content, page["title"])
            # elif self.Args.Generator.Maya.Action:
            #     mayaConvert(self.Args, content, page["title"])

            if pageNo != len(pages) - 1:
                time.sleep(30)


if __name__ == "__main__":
    VoiceGenerator().generation()