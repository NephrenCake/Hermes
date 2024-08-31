from typing import Dict, Union, Optional

from vllm.coinference.coinference import CoInference
from vllm.coinference.apps.single_request import SingleRequest
from vllm.coinference.apps.factool_code import FactoolCode
from vllm.coinference.apps.factool_kbqa import FactoolKBQA
from vllm.coinference.apps.factool_math import FactoolMath
from vllm.coinference.apps.react_fever import ReActFever
from vllm.coinference.apps.react_alfw import ReActAlfw
from vllm.coinference.apps.multiturn_conversations import MultiturnConversations
from vllm.coinference.apps.got_docmerge import GotDocMerge


AppLib: Dict[str, CoInference] = {
    None: SingleRequest,
    "factool_code": FactoolCode,
    "factool_kbqa": FactoolKBQA,
    "factool_math": FactoolMath,
    "react_fever": ReActFever,
    "react_alfw": ReActAlfw,
    "multiturn_conversations": MultiturnConversations,
    "got_docmerge": GotDocMerge,
}

def create_coinference(
    app_name: Union[None, str],
    coinf_id: str, 
    arrival_time: float,
    coinference_info_dict: Optional[Dict]
    ) -> CoInference:
    if app_name not in AppLib:
        raise NameError("Unrecognized app_name: %s", app_name)
    else:
        return AppLib[app_name](app_name, coinf_id, arrival_time, coinference_info_dict)
