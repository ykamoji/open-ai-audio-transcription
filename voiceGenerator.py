import os
import json
import yaml
import time
import argparse
from utils import createCache
from Emotions.sanitise import sanitise
from Emotions.emotions import addEmotions
from utils import CustomObject, get_yaml_loader, updateCache
from Generator.OpenAI import convert as openAIConvert
from Generator.Maya import convert as mayaConvert


class VoiceGenerator:

    def __init__(self):

        parser = argparse.ArgumentParser(description="Initidate data")
        parser.add_argument("--config", type=str, default="Default", help="Configuration file")
        parser.add_argument("--step", type=str, default="0", help="Step definition")
        args = parser.parse_args()

        with open('default.yaml', 'r') as file:
            config = yaml.load(file, get_yaml_loader())

        x = json.dumps(config)
        self.Args = json.loads(x, object_hook=lambda d: CustomObject(**d))

        self.Args.Platform = args.config
        self.Args.Step = int(args.step)

        with open('contentCache.json') as f:
            self.CONTENT_CACHE = json.load(f)

        self.VOICE_CACHE = createCache('voiceCache.json')
        self.EMOTION_CACHE = createCache('emotionCache.json')

    def load_content(self):
        data = []
        for pages in self.CONTENT_CACHE:
            data.append({
                "title": pages,
                "content": self.CONTENT_CACHE[pages]['content'],
            })
        return data

    def generation(self):

        notebook_name = self.Args.Graph.NotebookName
        section_name = self.Args.Graph.SectionName

        print(f"Running voice generation for {notebook_name} {section_name}")

        pages = self.load_content()

        limit = len(pages)
        if self.Args.Generator.PageLimit:
            limit = self.Args.Generator.PageLimit

        if self.Args.Step == 1:
            contents_to_process = []
            update_voice_cache = False
            for pageNo, page in enumerate(pages[:limit]):
                if not self.VOICE_CACHE or page["title"] not in self.VOICE_CACHE:
                    contents_to_process.append(page)
                    update_voice_cache = True

            spell_checked_paragraphs = sanitise(self.Args, contents_to_process)
            for page in spell_checked_paragraphs:
                self.VOICE_CACHE[page["title"]] = page['content']

            if update_voice_cache: updateCache('voiceCache.json', self.VOICE_CACHE)

            print(f"Spell check and grammar verified !")

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


if __name__ == "__main__":
    VoiceGenerator().generation()