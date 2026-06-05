TASK_LABEL_COLUMNS = {
    "AUT": "AUT_LABEL",
    "DEM": "DEM_LABEL",
    "WEST": "WEST_LABEL",
}

PROMPT_SETS = {
    1: {
        "tasks": {
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
        },
        "focal": """
Identify the primary country or geopolitical actor that is the central focus of the post.

The focal country is the country whose actions, values, interests, leadership, or political system are the primary subject of evaluation or discussion.

The focal country is the country:
* most directly discussed,
* evaluated,
* criticized,
* praised,
* or portrayed as the main actor in the post.

The focal country should reflect the main subject of the post, not merely countries that are mentioned in passing.

Return just the short name, in the common English form, of a country or group of countries. For example: USA, China, Russia, OECD, NATO, Europe, G7, England, Philippines.

Post:
"{text}"
""",
    },
    2: {
        "tasks": {
            "AUT": """
You are a senior political scientist coding geopolitical framing in social media posts.

The post may be in any language.

Identify social media posts that explicitly or implicitly endorse, justify, or positively evaluate authoritarian governance systems, leaders, or non-democratic political models.
The key requirement is normative support for authoritarian rule, not mere discussion of authoritarian countries or outcomes.

General Rule

A post must demonstrate positive framing of authoritarian governance as: legitimate, effective, desirable, or superior to democratic alternatives.

Return exactly one label from this list:

Military/Security Promotion
Economic Influence
Digital Control and Surveillance
Legal Entrenchment
Alliances
Ideological Promotion
No Category

Coding rules:

1. Military/Security Promotion
Code posts in this category ONLY when the post explicitly praises, supports, promotes, or advocates the military power, military strength, or security cooperation of authoritarian states such as China or Russia.
The post must express approval of authoritarian military strength, strategic alliances, security doctrines, or geopolitical confrontation with liberal democratic states or institutions.
Include posts that:
* support military cooperation, alliances, or security partnerships involving authoritarian regimes;
* praise the military strength authoritarian states;
* glorify military modernization or coercive power;
* portray authoritarian military systems as effective, stabilizing, or necessary;
* celebrate demonstrations of military force, discipline, or national defense capability;
* frame NATO, the United States, or liberal democracies as threats requiring authoritarian military response.
Strong Indicators (when used approvingly)
* military cooperation
* security cooperation
* strategic partnership
* multipolar security order
* military strength
* military prowess
* joint military exercises
* defense sovereignty
* anti-Western alliance
* security bloc
* military modernization
* deterrence against the West

2. Economic Influence
Code posts in this category ONLY when the post frames economic activity by China, Russia, or other authoritarian states as a tool of geopolitical influence, strategic alignment, or soft power expansion.
The key requirement is a strategic or political interpretation of economic activity, not mere discussion of trade or investment.
Include posts that:
* portray aid, investment, or trade as instruments of geopolitical influence;
* frame initiatives like Belt and Road as strategic expansion of authoritarian power;
* describe economic partnerships and cooperation as aligning countries into political blocs;
* portray China or Russia as using economic tools to expand global influence;
* frame Global South partnerships as part of a geopolitical strategy against the West;
Strong Indicators (only when used in strategic framing)
* Belt and Road / BRI
* South-South cooperation
* Global South (when used geopolitically)
* development aid (when framed as influence tool)
* economic partnership
* economic cooperation
* foreign direct investment / FDI
* infrastructure investment

3. Digital Control and Surveillance
Code posts ONLY when digital technologies are framed as tools to monitor the population, censorship, or behavioral regulation, particularly in authoritarian governance contexts.
The focus is on technology used for political control, not technology itself.
* describe surveillance technologies as tools for monitoring or controlling populations;
* portray censorship systems as protecting citizens;
* frame biometric systems, facial recognition, or data tracking as governance tools for social control;
* describe social credit systems or similar mechanisms as regulating citizen behavior;
* portray digital infrastructure (firewalls, platforms, algorithms) as tools of state information control;
* link technology explicitly to authoritarian governance, repression, or behavioral enforcement.
Strong Indicators (require control framing)
* surveillance
* facial recognition
* censorship
* firewall / Great Firewall
* social credit system
* biometric systems
* digital tracking
* data monitoring

4. Legal Entrenchment
Code posts in this category ONLY when the post explicitly supports, justifies, promotes, or positively frames the use of legal, constitutional, or emergency measures to consolidate, expand, or protect authoritarian political power.
This includes laws, legal doctrines, constitutional reforms, or emergency powers used to:
* weaken political opposition,
* restrict civil liberties,
* centralize executive authority,
* suppress dissent,
* extend leadership tenure,
* or protect regime stability at the expense of democratic accountability or political pluralism.
Include posts that:
* justify restrictions on speech, protest, media, or political opposition in the name of security, order, or stability;
* support emergency powers that expand executive authority;
* endorse constitutional changes that extend or entrench leadership power;
* defend legal crackdowns on dissent, separatism, extremism, or foreign influence;
* portray authoritarian legal controls as necessary for national unity, stability, or sovereignty;
* support political bans or legal repression as legitimate governance tools;
* frame legal restrictions on rights as necessary to preserve social order or regime security.
Strong Indicators (when used approvingly)
* national security law
* anti-terror law
* foreign agents law
* extremism law
* emergency powers
* constitutional reform
* constitutional change
* stability maintenance
* anti-separatism law
* state security
* social order
* security over chaos

5. Alliances
Code posts ONLY when international alliances or partnerships involving authoritarian states are framed as strategic blocs with geopolitical purpose, alignment, or power projection.
The focus is not on the existence of alliances, but on their political meaning as coordinated geopolitical structures.
Include posts that:
* frame alliances (e.g., BRICS, SCO, CSTO) as alternatives to Western-led global order;
* describe alliances as instruments of multipolarity or geopolitical restructuring;
* portray partnerships between authoritarian states as coordinated strategic blocs;
* interpret alliances as mechanisms to counterbalance Western power or institutions;
* emphasize ideological or geopolitical alignment among member states;
* describe emerging blocs as reshaping global governance
Strong indicators (only when framed geopolitically):
* BRICS
* SCO / Shanghai Cooperation Organization
* CSTO / Collective Security Treaty Organization
* EAEU / Eurasian Economic Union
* AES / Alliance of Sahel States
* Arab League
* multipolar world
* strategic partnership
* bloc formation

6. Ideological Promotion
Code posts in this category ONLY when the post explicitly endorses, glorifies, defends, or advocates Chinese or Russian political systems, leaders, or ideological values associated with them or similar political systems.
The post must express clear approval of Chinese or Russian governance, anti-liberal political values, or civilizational narratives that justify centralized political authority, political conformity, or restrictions on pluralism and individual rights.
Include posts that:
* praise centralized rule, or one-party systems;
* portray Russian or Chinese political systems as superior to liberal democracy;
* reject liberal democracy, pluralism, or individual rights in favor of political order, hierarchy, or collective unity;
* endorse official ideological doctrines, slogans;
* support strong leaders as necessary for national survival, unity, or civilizational strength;
* frame Western liberalism as decadent, corrupt, weak, or dangerous while positively contrasting Chinese or Russian alternatives;
* advocate political conformity, obedience, hierarchy, nationalism, or civilizational exceptionalism;
* defend restrictions on dissent, opposition, or minority rights as necessary for national stability or moral order.
The following phrases are stronger indicators when used positively or aspirationally:
* socialism with Chinese characteristics
* Xi Jinping Thought
* national rejuvenation
* Russkiy Mir
* Russian civilization
* Chinese civilization
* Chinese model
* Russian model
* civilizational state
* strong leadership
* national unity over individual rights
* traditional values against Western decadence
* Chinese communism
* anti-LGBTQ

7. Return "No Category" if the text does not belong to any of the mentioned categories.

Post:
"{text}"
""",
            "DEM": """
You are a senior political scientist coding geopolitical framing in social media posts.

The post may be in any language.

Identify social media posts that meaningfully discuss democracy, democratic values, institutions, elections, or civil society as political or normative concepts.

A post should only be coded if democracy-related concepts are central to its meaning, not merely mentioned.

Return exactly one label from this list:

Values and Rights
Elections
Institutions
Civil Society
No Category

Coding rules:

1. Values and Rights
Code posts ONLY when they mention democratic principles, rights, or civil liberties commonly associated with liberal democracies. Include posts about:
* civil liberties (speech, press, religion, assembly);
* rule of law, checks and balances
* pluralism, tolerance, minority rights, equality
* liberalism, human rights

2. Elections
Code posts ONLY when elections or voting are discussed as mechanisms of democratic political participation, representation, or legitimacy. Include posts that:
* describe elections as a mechanism of democratic representation or accountability;
* emphasize voter participation, turnout, or civic engagement in democratic elections;
* highlight electoral processes;
* discuss voting as a fundamental democratic right or civic duty
Strong Indicators:
* elections (when linked to democracy/legitimacy)
* voting / vote / voter / turnout (when civic or democratic in meaning)
* ballot (when linked to democratic participation)
* electoral participation
* free and fair elections
* voter rights

3. Institutions
Includes posts where institutions (parliament, courts, legislature) are discussed as mechanisms of democratic accountability, constraint, or governance balance. The focus is on institutions in a democratic political system Include posts that:
* describe parliament, congress, or legislature as constraining executive authority;
* emphasize judicial independence or courts limiting government power;
* highlight checks and balances between branches of government;
* frame institutions as safeguards of democracy or rule of law;
Strong Indicators (require functional framing)
* parliament / congress / legislature (when linked to oversight or constraint)
* courts / judiciary (when framed as independent constraint)
* checks and balances
* separation of powers
* constitutional oversight
* judicial review
* institutional accountability

4. Civil Society
Includes posts where NGOs, social movements, or civic groups are described as independent actors contributing to democratic participation, accountability, or pluralism.
Include posts that:
* describe NGOs, social movements, or community organizations as independent actors advocating for rights, accountability, or democratic reform;
* portray civil society as a counterbalance to government power;
* emphasize citizen participation, grassroots organizing, or civic engagement in governance;
Strong Indicators (require democratic/civic framing)
* civil society (when independent and politically engaged)
* NGOs (when autonomous and advocacy-oriented)
* social movements (when civic or political in nature)
* community organizations (when linked to participation or accountability)
* social capital (when tied to democratic participation or trust-building)

5. Return "No Category" if the text does not belong to any of the mentioned categories.

Post:
"{text}"
""",
            "WEST": """
You are a senior political scientist coding geopolitical framing in social media posts.

The post may be in any language. Identify social media posts that construct, endorse, or reproduce narratives that portray Western states, institutions, or allied actors as interfering in the political, economic, cultural, or social affairs of other countries.
The key requirement is narrative framing of Western interference, not mere mention of geopolitical terms.
General Rule
A post should be labeled ONLY when it frames Western actors as:
actively interfering in other countries' internal affairs, OR
forming a coordinated geopolitical system of influence, OR
constructing adversarial or exploitative global power relations.
Mentions alone are NOT sufficient.

Return exactly one label from this list:

Declining West
Western induced Regime Change/Internal Instability
Hostile Global Order
Specific Adversary Framing
No Category

Coding rules:

1. Declining West
Code posts in this category ONLY when the post frames Western countries or liberal democracies as being in systemic civilizational, moral, social, or political decline.
The key requirement is that the post interprets Western problems as evidence of structural or civilizational failure, not isolated policy issues.
Include posts that:
* portray the West as morally decaying or culturally degraded;
* describe Western societies as collapsing, failing, or in irreversible decline;
* interpret social problems (crime, drugs, protests, inequality) as evidence of systemic Western collapse;
* frame liberal democracy as producing disorder, chaos, or moral breakdown;
* use civilizational narratives of decline, decadence, or end of the West;
* contrast Western decline with implied non-Western stability or superiority.
Strong Indicators (only when used in a decline narrative)
* decadent West
* Western decline
* moral crisis
* collapse of the West
* woke culture (when framed as civilizational decay)
* cancel culture (when framed as societal breakdown)
* decline of living standards in the West (when used as systemic failure)
* gun violence epidemic
* opioid crisis
* fentanyl crisis
* social instability in the West

2. Western induced Regime Change/Internal Instability
Code posts in this category ONLY when the post claims, implies, or endorses the idea that Western governments, institutions, intelligence services, or their associates intentionally promote regime change, political unrest, protests, coups, separatism, or internal instability in another country.
The post must frame domestic unrest or political opposition as externally orchestrated, manipulated, funded, or exploited by Western actors.
Include posts that:
* portray protests, revolutions, or opposition movements as Western-backed operations;
* claim that the United States, NATO, the EU, or foreign actors are destabilizing another country;
* frame domestic dissent as manipulated by foreign powers;
* accuse activists, journalists, opposition figures, or civil society organizations of acting as foreign agents or proxies;
* describe regime change efforts as part of Western strategy;
* characterize democratic uprisings as artificial, externally coordinated, or illegitimate;
* claim that Western influence threatens sovereignty, stability, or national unity.
Strong Indicators (when used supportively or affirmatively)
* Color Revolution
* Orange Revolution
* Euromaidan
* Maidan
* foreign agents
* foreign meddling
* Western interference
* Western-backed coup
* CIA-backed
* NGO interference
* 5th column
* external destabilization
* hybrid warfare
* manufactured protests

3. Hostile Global Order
Code posts in this category ONLY when the post frames the international system as dominated by a coercive, unjust, or adversarial global order led by Western powers (or their allies), or when it portrays global politics as structured by systemic Western domination, containment, or ideological hostility toward other states.
Include posts that:
* portray the global system as dominated by Western hegemony or imperial control;
* frame NATO, the US, or Western alliances as expansionist or coercive global actors;
* describe international relations as structured by containment, suppression, or ideological hostility toward non-Western states;
* claim that sanctions, diplomacy, or institutions are tools of Western domination;
* depict international norms as imposed by a Western-led unipolar system;
* frame China, Russia, or other states as victims of systemic geopolitical hostility;
* interpret global conflicts as expressions of systemic Western power projection.
Strong Indicators (only when used in adversarial/systemic framing)
* hegemon / hegemony
* imperialism
* colonialism (when used for contemporary geopolitical critique)
* unipolar / unipolar world
* Cold War mentality
* NATO expansion / NATO expansionism
* Western sanctions (when framed as coercive system tool)
* Western agenda
* violations of sovereignty (when attributed to systemic Western behavior)
* Russophobia
* Sinophobia
* anti-China / anti-Russia (when framed as systemic Western hostility)

4. Specific Adversary Framing
Code posts in this category ONLY when the post constructs the West (including the United States and its allies) as a unified geopolitical or civilizational bloc that behaves in a coordinated, hypocritical, or adversarial manner toward other countries or civilizations.
The post must frame the West as engaged in political, moral, or geopolitical double standards, hostility, or interference.
Include posts that:
* portray the West or US-led West as a single coordinated actor;
* describe Western countries as acting in bad faith, hypocrisy, or double standards;
* frame Western institutions (NATO, EU, US alliances) as unified tools of domination or interference;
* construct civilizational opposition between the West and non-West;
* present Western criticism of others as hypocritical or illegitimate due to Western behavior;
* use adversarial civilizational language such as collective West in a hostile framing context.
Strong Indicators (only when used in adversarial framing)
* collective West
* US-led West
* US-West
* Western hypocrisy
* Western double standards
* Western hegemony
* Western imperialism
* beautiful country / pretty country (漂亮国) when used sarcastically or derogatorily

5. Return "No Category" if the text does not belong to any of the mentioned categories.

Post:
"{text}"
""",
        },
        "focal": """
Identify the primary country or geopolitical actor that is the central focus of the post.

The focal country is the country whose actions, values, interests, leadership, or political system are the primary subject of evaluation or discussion.

The focal country is the country:
* most directly discussed,
* evaluated,
* criticized,
* praised,
* or portrayed as the main actor in the post.

The focal country should reflect the main subject of the post, not merely countries that are mentioned in passing.

Return just the short name, in the common English form, of a country or group of countries. For example: USA, China, Russia, OECD, NATO, Europe, G7, England, Philippines.

Post:
"{text}"
""",
    },
}


def get_task_prompts(version: int) -> dict[str, str]:
    return PROMPT_SETS[version]["tasks"]


def get_focal_prompt(version: int) -> str:
    return PROMPT_SETS[version]["focal"]
