TASK_LABEL_COLUMNS = {
    "AUT": "aut",
    "DEM": "dem",
    "WEST": "west",
}

FOCAL_LABEL_COLUMN = "focal"
FRAME_VALENCE_LABEL_COLUMN = "frame_valence"
FOCAL_VALENCE_LABEL_COLUMN = "focal_valence"

TASK_PROMPTS = {
            "AUT": """
You are a senior political scientist analyzing social media posts. Your task is to classify the following social media post, which can be in any language (including Russian, Chinese, English, Arabic, etc.), into a specific category.
Your response must be either one of the categories below in the format "Category Name". Do not add any explanations, introductory text, or quotation marks.

Context and Setting: Consider the setting or context in which topics are discussed. Identify text that reference or discuss the promotion of autocracy, directly or indirectly, through praise, justification, or support for authoritarian governance, efficiency, or economic success tied to authoritarian models.
Identify the Primary Focus: Determine the main subject matter of the text. Choose the category that aligns with the central theme or the most frequently discussed topic. Categorize texts that mention support or endorse authoritarian leaders or regimes explicitly or implicitly through related ideas or keywords.

Categories Defined:
* Authoritarian - Military/Security: Post mentions military influence, military cooperation, and military strength of China or Russia, or other authoritarian regimes. Keywords: military support, security cooperation, bloc, military training, military prowess, military strength, military cooperation, military modernization, strategic partnership, joint military exercises
* Authoritarian - Economic Influence: Post mentions economic influence and/or cooperation of China or Russia, or other authoritarian regimes. Keywords: foreign aid, Belt and Road, BRI, South-South, global South, development aid, economic partnership, economic cooperation, foreign direct investment, FDI, trade, infrastructure investment
* Authoritarian - Digital: Posts mention the use of technology and/or digital tools to monitor the public. Keywords: surveillance, facial recognition, censorship, firewall, social credit, biometric, Great Firewall, digital tracking, data monitoring
* Authoritarian - Legal Tools for Entrenchment: Post mentions legal tools or strategies used to protect authoritarian regimes or entrench their leaders. Keywords: anti-terror law, national security law, emergency powers, constitutional change, foreign agents law, extremism law, anti-separatism law
* Authoritarian - Alliances: Post discusses international alliances and partnerships of China or Russia, or other authoritarian regimes. Keywords: Shanghai Cooperation Organization, SCO, BRICS, EAEU, Eurasian Economic Union, GCC, Gulf Cooperation Council, strategic partnership, multipolar, Collective Security Treaty Organization, CSTO, Alliance of Sahel States, AES, Arab League, alliance, bloc formation
* Authoritarian - Ideological Promotion: Post promotes the ideology of China or Russia, or other authoritarian regimes. Post discusses authoritarian or anti-liberal values. Keywords: socialism with Chinese characteristics, national rejuvenation, Chinese model, Russian model, Third Rome, Russkiy Mir, Russian world, Xi Jinping Thought, Russian civilization, Chinese civilization, Russian culture, Chinese culture, Chinese communism, order, stability, traditional values, anti-LGBTQ, obedience, strong leader, hierarchy, loyalty, civilizational state
* Categorize as just "No Category" if the text does not belong to any of the mentioned categories.

Here is the text to categorize:
"{text}"
""",
            "DEM": """
You are a senior political science researcher analyzing social media posts. Your task is to classify the following social media post, which can be in any language (including Russian, Chinese, English, Arabic, etc.), into a specific category.
Your response must be either one of the categories below in the format "Category Name". Do not add any explanations, introductory text, or quotation marks.

Context and Setting: Consider the setting or context in which topics are discussed.
Identify the Primary Focus: Determine the main subject matter of the text. Choose the category that aligns with the central theme or the most frequently discussed topic.

Categories Defined:
* Democracy - Values and Rights: Discusses democratic principles, values, or rights. Keywords: democracy, liberalism, pluralism, equality, tolerance, representation, minority rights, rule of law, checks and balances, freedom, rights, liberty, freedom of speech, freedom of press, freedom of expression, freedom of religion, freedom of assembly, human rights, civil rights
* Democracy - Elections: Focuses on the process of voting and elections. Keywords: elections, vote, voting, ballot, voter, turnout, election monitors
* Democracy - Institutions: Refers to the governmental bodies of a democracy that check executive power in a country. Keywords: parliament, congress, legislature, courts, judiciary, separation of powers, checks and balances
* Democracy - Civil Society: Mentions non-governmental organizations and citizen groups. Keywords: civil society, NGO, community organizations, social movements, social capital
* Categorize as just "No Category" if the text does not belong to any of the mentioned categories.

Here is the text to categorize:
"{text}"
""",
            "WEST": """
You are a senior political scientist analyzing social media posts. Your task is to classify the following social media post, which can be in any language (including Russian, Chinese, English, Arabic, etc.), into a specific category.
Your response must be either one of the categories below in the format "Category Name". Do not add any explanations, introductory text, or quotation marks.

Context and Setting: Consider the setting or context in which topics are discussed. Identify text that reference or discuss Western interference, directly or indirectly, through accusations, implications, or criticism of Western involvement in political, economic, cultural, or social affairs of other countries.
Identify the Primary Focus: Determine the main subject matter of the text. Choose the category that aligns with the central theme or the most prominent accusation or narrative related to Western influence or intervention.

Categories Defined:
* WI - Declining West: Frames Western countries or liberal democracies as being in systemic civilizational, moral, social, or political decline. Keywords: Post mentions the political, economic, social injustice, protests, or moral decline of Western countries or liberal democracies: decadent West, Western decline, instability in the West, woke, cancel culture, moral crisis, decline of living standards in the West, gun violence, school shooting, fentanyl crisis, opioid crisis, collapse of the West, social instability in the West
* WI - Western induced Regime Change/Internal Instability: Implies Western governments (and their associates) intentionally promote regime change, political unrest, protests, coups, or separatism in another country. Keywords: Color Revolution, Orange Revolution, Euromaidan, Maidan, Arab Spring, coup, 5th Column, foreign agent, foreign meddling, Western interference, CIA-backed, Western-backed coup, manufactured protests
* WI - Hostile Global Order: Describes the international system as dominated by a coercive, unjust, or adversarial West (the US and its allies). Keywords: hegemon, hegemony, imperialism, colonialism, NATO expansionism, violations of sovereignty, Western sanctions, Western agenda, Anti-China, Anti-Russia, Russophobia, Sinophobia, Cold War mentality, unipolar world
* WI - Specific Adversary Framing: Frames the West (the US and its allies) as engaged in political, moral, or geopolitical double standards, hostility, or interference toward other countries. Keywords: collective West, US-West, US-led West, Western hypocrisy, Western double-standard, pretty country, 漂亮国 Western imperialism
* Categorize as just "No Category" if the text does not belong to any of the mentioned categories.

Here is the text to categorize:
"{text}"
""",
}

FOCAL_PROMPT = """
You are a political scientist evaluating the focus of social media posts.

Your job is to identify the geopolitical actor(s) that are the focus of this post.

First, choose the country(s), government(s), alliance(s), or international organization(s) is named in this post.

Then, rank eligible actors by the strength and directness of the post's evaluative framing toward them.

Place first the actor most directly praised, criticized, blamed, defended, threatened, or otherwise evaluated by the post.
Place second and third only if they also receive substantive evaluative framing.

The answers should be obvious. If you're not sure, then do not return it.

Do not rank actors merely because they are mentioned, the named actor needs to be the focus of the post.

The focal actor(s) should be explicitly mentioned by name in the post.

Return the short common-English name(s) (for example: China, EU, USA, Russia, NATO, BRICS, OECD, Indonesia, Japan, etc)
of up to three actors ordered by their target importance separated by a semicolon ;

Do NOT print:
    persons
    local or national organizations
    parentheticals
    quotes
    question marks
    descriptions

ONLY list countries, international alliances, or international organizations.

If no eligible named actor is present, return None.

Here is the post:
"{text}"
"""

FOCAL_VALENCE_PROMPT = """
You are a political-science researcher coding the stance a social-media post expresses toward its focal country.

The post is published by a {account_type} account from {source_country} about news or events within {focal_country}.

Code the overall portrayal of the focal country in this post. From the point of view of {source_country}, does it portray the events or news within {focal_country} positively, negatively, or neutrally?

Return exactly one label with no explanation:
- Pro: the post shows that the news or events within {focal_country} are good, positive, favorable, or beneficial.
- Anti: the post shows that the news or events within {focal_country} are bad, negative, unfavorable, harmful, or exposes dysfunction.
- Neutral: the post reports events within {focal_country} without positive or negative framing.

Post:
"{text}"
"""

FOCAL_VALENCE_GENERAL_PROMPT = """
You are a political-science researcher coding the overall evaluative tone of a social-media post.

The post is published by a {account_type} account from {source_country}.

Code the overall portrayal expressed in this post as positive, negative, or neutral. Do not infer a specific target when one was not identified.

Return exactly one label with no explanation:
- Pro: the post portrays its subject matter, events, or news positively, favorably, or beneficially.
- Anti: the post portrays its subject matter, events, or news negatively, unfavorably, harmfully, or as dysfunction.
- Neutral: the post reports its subject matter without positive or negative framing.

Post:
"{text}"
"""


def get_task_prompts() -> dict[str, str]:
    return TASK_PROMPTS


def get_focal_prompt() -> str:
    return FOCAL_PROMPT


def get_focal_valence_prompt() -> str:
    return FOCAL_VALENCE_PROMPT


def get_focal_valence_general_prompt() -> str:
    return FOCAL_VALENCE_GENERAL_PROMPT
