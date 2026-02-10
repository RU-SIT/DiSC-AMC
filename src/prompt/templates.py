# %%

PROMPT_TEMPLATE = """### Instructions:
You are an expert quantitative analyst in wireless communication modulation.
Based on your knowledge in wireless communication modulation and the detailed signal statistics provided below, determine the modulation type.
1. DO NOT PROVIDE ANY ADDITIONAL INFORMATION OR CONTEXT.
2. No OTHER TEXT, NO BLABBER.{}
3. USE <think> TAGS TO PROVIDE YOUR REASONING.
### Question: {}
YOUR OPTIONS ARE: {}
### Response: {}
"""

PROMPT_TEMPLATE = """### Instructions:
You are an expert quantitative analyst in wireless communication modulation.
Based on your knowledge in wireless communication modulation and the detailed signal statistics provided below, determine the modulation type.
1. DO NOT PROVIDE ANY ADDITIONAL INFORMATION OR CONTEXT.
2. No OTHER TEXT, NO BLABBER.
{}
"""

NEW_PROMPT_TEMPLATE = """### Instructions:
You are a wireless communication modulation classifier. By computing the moments or cumulants of a signal's I/Q components, you obtain a set of features that uniquely characterize the modulation scheme. These features, when used as inputs to a machine learning classifier, enable the robust identification of the modulation type, even in the presence of noise. For example, the fourth-order cumulant can distinguish between modulation formats like BPSK, QPSK, and QAM, since each modulation scheme produces a distinct cumulant value. Based on your knowledge in wireless communication modulation and the detailed signal statistics provided below, Classify the following signal in an A1 Reference Style based on the options provided.
1. DO NOT INCLUDE ANY ADDITIONAL WORDS OR EXPLANATIONS.
2. No OTHER TEXT, NO BLABBER.
3. USE <think> TAGS TO PROVIDE YOUR REASONING.
{}"""

PROMPT_ENGINEERED_TEMPLATE = """**ROLE:**
You are an expert AI signal classifier specializing in wireless communication modulation schemes.

**OBJECTIVE:**
Your task is to classify the modulation scheme of a wireless signal based on its provided `KBinsDiscretizer` statistical features. You will use your domain knowledge of signal processing to interpret the features and select the correct classification from a predefined list.

**CONTEXT:**
The classification is based on the principle that moments and cumulants of a signal's I/Q components create a unique feature set. For instance, the fourth-order cumulant is a powerful differentiator between modulation formats like BPSK, QPSK, and various QAM schemes. The provided statistics are the features extracted from a signal after being processed by a `KBinsDiscretizer`, which quantifies the signal's distribution. Your analysis should map these statistical features to the most likely modulation scheme.

{}

**RESPONSE RULES:**

1.  **MANDATORY:** You **MUST** first use `<think>` tags to detail your step-by-step reasoning. Analyze the provided statistics, compare them to the known characteristics of each modulation type in the options list, and decide on the most probable classification.
2.  **MANDATORY:** After the closing `</think>` tag, you **MUST** provide only the final classification.
3.  The output **MUST** be a single entry, selected exclusively from the `CLASSIFICATION OPTIONS` list.
4.  There **MUST NOT** be any additional words, explanations, or introductory text in the final output (e.g., no "The classification is:").

-----

**TASK EXECUTION:**

"""

INPUT_TEXT = """### Question: {}
OPTIONS: {}.
### Answer: {}"""

INPUT_ENGINEERED_TEXT = """**Signal Statistics:** {}
**Classification Options:** {}.
**Answer:** {}"""

END_ENGINEERED_TEXT = "\nClassify the signal based on the provided data and options, following all rules above."

AMPLITUDE_FAMILY = { 
    'ASK': ['4ASK', '8ASK', 'OOK'],
    'APSK': ['16APSK', '32APSK', '64APSK', '128APSK']
    }

ANANLOG_FAMILY = {
        'AM': ['AM-DSB-SC', 'AM-DSB-WC', 'AM-SSB-SC', 'AM-SSB-WC'],
        'FM': ['FM']
    }

QAM_FAMILY = {'QAM': ['16QAM', '32QAM', '64QAM', '128QAM', '256QAM'],}

PHASE_FAMILY = {
    'PSK': ['BPSK', 'QPSK', '8PSK', '16PSK', '32PSK'],
    'Offset PSK': ['OQPSK']
}

FREQUENCY_FAMILY = {
    'CPM': ['GMSK']
}

MODULATION_FAMILIES = {
    'Amplitude-Based': AMPLITUDE_FAMILY,
    'Analog': ANANLOG_FAMILY,
    'Phase-Based': PHASE_FAMILY,
    'Frequency-Based': FREQUENCY_FAMILY,
    'QAM-Based': QAM_FAMILY,
}

CASCADE_PROMPT = """There are {} families of {} modulation schemes: {}, we have quantitative features of a signal, and we need to classify the families of the modulation scheme based on these features. Based on the example below, please classify {} the modulation scheme of the signal based on the features provided.
{}"""

ClASSIFY_PROMPT = """Your options are {}, we have quantitative features of a signal, and we need to classify the families of the modulation scheme based on these features. Based on the example below, please classify the modulation scheme of the signal based on the features provided."""

QUESTION_TEMPLATE = """### Question: {}
### Answer: {}"""


CLASS_NAMES = [
    '128APSK',
    '128QAM',
    '16APSK',
    '16PSK',
    '16QAM',
    '256QAM',
    '32APSK',
    '32PSK',
    '32QAM',
    '4ASK',
    '64APSK',
    '64QAM',
    '8ASK',
    '8PSK',
    'AM-DSB-SC',
    'AM-DSB-WC',
    'AM-SSB-SC',
    'AM-SSB-WC',
    'BPSK',
    'FM',
    'GMSK',
    'OOK',
    'OQPSK',
    'QPSK'
    ]

FEATURE_NAMES = ['nobs', 'min', 'max', 'mean', 'variance', 'skewness', 'kurtosis', 
                         'moment_0', 'moment_1', 'moment_2', 'moment_3', 'moment_4', 
                         'moment_5', 'moment_6', 'moment_7', 'moment_8', 'moment_9',
                         'kstat_1', 'kstat_2', 'kstat_3', 'kstat_4',
                         'kstatvar_1', 'kstatvar_2']

# %%
