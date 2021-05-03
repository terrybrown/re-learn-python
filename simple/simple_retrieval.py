from urllib.request import urlopen

def get_words():
    story = urlopen('http://sixty-north.com/c/t.txt')
    storyWords = []

    # Original way shown in tutorial
    # for line in story:
    #     lineWords = line.decode('utf8').split()
    #     for word in lineWords:
    #         storyWords.append(word)

    # Better way? (we shall find out)
    storyWords = story.read().decode('utf8').split();

    story.close()

    return storyWords
