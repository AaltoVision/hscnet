from .seven_scenes import SevenScenes
from .twelve_scenes import TwelveScenes
from .cambridge_landmarks import Cambridge

def get_dataset(name):

    return {
            '7S' : SevenScenes,
            '12S' : TwelveScenes,
            'Cambridge' : Cambridge         
           }[name]
