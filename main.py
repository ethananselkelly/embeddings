import openai
from openai.embeddings_utils import get_embedding, cosine_similarity
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

paragraphs = [
    "Shortly after arriving, my body began shutting down. I struggled with basic functions like swallowing and breathing. I had my first seizure of the day. Then I stopped breathing entirely. As the doctors hurried to supply me with oxygen, they also decided the local hospital was unequipped to handle the situation and ordered a helicopter to fly me to a larger hospital in Cincinnati.",
    "I was rolled out of the emergency room doors and toward the helipad across the street. The stretcher rattled on a bumpy sidewalk as one nurse pushed me along while another pumped each breath into me by hand. My mother, who had arrived at the hospital a few moments before, climbed into the helicopter beside me. I remained unconscious and unable to breathe on my own as she held my hand during the flight.",
    "While my mother rode with me in the helicopter, my father went home to check on my brother and sister and break the news to them. He choked back tears as he explained to my sister that he would miss her eighth-grade graduation ceremony that night. After passing my siblings off to family and friends, he drove to Cincinnati to meet my mother.",
    "When my mom and I landed on the roof of the hospital, a team of nearly twenty doctors and nurses sprinted onto the helipad and wheeled me into the trauma unit. By this time, the swelling in my brain had become so severe that I was having repeated post-traumatic seizures. My broken bones needed to be fixed, but I was in no condition to undergo surgery. After yet another seizure—my third of the day—I was put into a medically induced coma and placed on a ventilator.",
    "My parents were no strangers to this hospital. Ten years earlier, they had entered the same building on the ground floor after my sister was diagnosed with leukemia at age three. I was five at the time. My brother was just six months old. After two and a half years of chemotherapy treatments, spinal taps, and bone marrow biopsies, my little sister finally walked out of the hospital happy, healthy, and cancer free. And now, after ten years of normal life, my parents found themselves back in the same place with a different child.",
    "While I slipped into a coma, the hospital sent a priest and a social worker to comfort my parents. It was the same priest who had met with them a decade earlier on the evening they found out my sister had cancer.",
    "As day faded into night, a series of machines kept me alive. My parents slept restlessly on a hospital mattress—one moment they would collapse from fatigue, the next they would be wide awake with worry. My mother would tell me later, “It was one of the worst nights I’ve ever had.”",
    "Mercifully, by the next morning my breathing had rebounded to the point where the doctors felt comfortable releasing me from the coma. When I finally regained consciousness, I discovered that I had lost my ability to smell. As a test, a nurse asked me to blow my nose and sniff an apple juice box. My sense of smell returned, but—to everyone’s surprise—the act of blowing my nose forced air through the fractures in my eye socket and pushed my left eye outward. My eyeball bulged out of the socket, held precariously in place by my eyelid and the optic nerve attaching my eye to my brain.",
    "The ophthalmologist said my eye would gradually slide back into place as the air seeped out, but it was hard to tell how long this would take. I was scheduled for surgery one week later, which would allow me some additional time to heal. I looked like I had been on the wrong end of a boxing match, but I was cleared to leave the hospital. I returned home with a broken nose, half a dozen facial fractures, and a bulging left eye.",
    "The following months were hard. It felt like everything in my life was on pause. I had double vision for weeks; I literally couldn’t see straight. It took more than a month, but my eyeball did eventually return to its normal location. Between the seizures and my vision problems, it was eight months before I could drive a car again. At physical therapy, I practiced basic motor patterns like walking in a straight line. I was determined not to let my injury get me down, but there were more than a few moments when I felt depressed and overwhelmed."
]

vector_dictionary = []

for paragraph in paragraphs:
    response = openai.Embedding.create(
        input=[paragraph],
        model='text-embedding-ada-002'
    )
    vector_dictionary.append(response['data'][0]['embedding'])

while True:
    search_term = input('search term: ')

    search_term_vector = openai.Embedding.create(
        input=[search_term],
        model='text-embedding-ada-002'
    )['data'][0]['embedding']

    dot_array = []

    for i, vector in enumerate(vector_dictionary):
        dot_product = np.dot(search_term_vector, vector)
        dot_array.append([dot_product, i])

    hit = sorted(dot_array, key=lambda x: x[0], reverse=True)[0]
    hit1 = sorted(dot_array, key=lambda x: x[0], reverse=True)[1]
    hit2 = sorted(dot_array, key=lambda x: x[0], reverse=True)[2]
    print("Result 1:\n" + paragraphs[hit[1]])
    print("Result 2:\n" + paragraphs[hit1[1]])
    print("Result 3:\n" + paragraphs[hit2[1]])
