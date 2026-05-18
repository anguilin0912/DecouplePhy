from .DecouplePhy.DecouplePhy import DecouplePhy
from .Knowledgebased.ExpertInvariants import ExpertInvariants

idss = [
    DecouplePhy,
    ExpertInvariants,
]

def get_all_iidss():
    return {ids._name: ids for ids in idss}
