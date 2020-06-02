import random
import numpy as np

from datasets.text_renderer.textrenderer.corpus.corpus import Corpus


class ChnCorpus(Corpus):
    def load(self):
        """
        Load one corpus file as one line , and get random {self.length} words as result
        """
        # self.load_corpus_path()
        # self.corpus_path = ['./data/corpus/txt_gen/book.txt', 
        #                     './data/corpus/txt_gen/1.txt', 
        #                     './data/corpus/txt_gen/num.txt', 
        #                     './data/corpus/txt_gen/simi_chars.txt',
        #                     './data/corpus/txt_gen/simi_chs.txt', 
        #                     './data/corpus/txt_gen/random_chinese_word.txt']
        # self.distrib = [0.2, 0.2, 0.1, 0.05, 0.1, 0.35] # book, book, num, simi chars, simi chinese, random_chinese
        self.corpus_path = ['./datasets/corpus/charset.txt']
        self.distrib = [1.0]
        
        #self.corpus.append('')
        for i, p in enumerate(self.corpus_path):
            print_end = '\n' if i == len(self.corpus_path) - 1 else '\r'
            print("Loading chn corpus: {}/{}".format(i + 1, len(self.corpus_path)), end=print_end)
            with open(p, encoding='utf-8') as f:
                data = f.readlines()

            lines = []
            for line in data:
                line_striped = line.strip()
                line_striped = line_striped.replace(" ", "")
                # line_striped = line_striped.replace('\u3000', '')
                # line_striped = line_striped.replace('&nbsp', '')
                # line_striped = line_striped.replace("\00", "")
                # line_striped = line_striped.replace("〗", "")
                # line_striped = line_striped.replace("〖", "")

                if line_striped != u'' and len(line.strip()) > 1:
                    lines.append(line_striped)

            # 所有行合并成一行
            # split_chars = [',', '，', '：', '-', ' ', ';', '。']
            split_chars = [',', ':', '-', ';', '。']
            splitchar = random.choice(split_chars)
            whole_line = splitchar.join(lines)

            # 在 crnn/libs/label_converter 中 encode 时还会进行过滤
            whole_line = ''.join(filter(lambda x: x in self.charsets, whole_line))

            if len(whole_line) > 20:
                #self.corpus.append(whole_line)
                #whole_line = whole_line.replace('   ', '')
                self.corpus.append(whole_line)


    def get_sample(self, img_index):
        # 每次 gen_word，随机选一个预料文件，随机获得长度为 word_length 的字符        
        line = np.random.choice(self.corpus, 1, p=self.distrib)[0]
        line += ' '
        # if ' ' in line:
        #     print('------- true ----------------', line.count(' '), len(line))

        list_line = list(line)
        random.shuffle(list_line)
        line = ''.join(list_line)
        while True:
            start = np.random.randint(0, len(line) - self.length)
            word = line[start:start + self.length]
            if not (word.startswith(' ') or word.endswith(' ')):
                break

        #print(word, len(word), img_index)
        
        # i = 0        
        # while len(word) > len(word.strip()) and i<4:            
        #     start = np.random.randint(0, len(line) - self.length)
        #     word = line[start:start + self.length]
        #     i += 1

        # if random.random() < 0.01:
        #     idx = random.randint(1, self.length-2)
        #     word = word.replace(word[idx], ' ')

        # if len(word.strip()) < len(word):
        #     start = np.random.randint(0, len(line) - self.length)
        #     word = line[start:start + self.length]

        return word
