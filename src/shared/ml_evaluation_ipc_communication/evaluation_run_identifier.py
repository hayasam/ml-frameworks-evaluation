class EvaluationRunIdentifier(object):
    def __init__(self, name: str, evaluation_type: str, challenge: str, lib_name: str, model_name: str):
        self.name = name
        self.evaluation_type = evaluation_type
        self.challenge = challenge
        self.library_name = lib_name
        self.model_name = model_name

    @classmethod
    def from_dict(cls, d: dict):
        return cls(name=d['name'], evaluation_type=d['evaluation_type'], challenge=d['challenge'], lib_name=d['library_name'], model_name=d['model_name'])

    @staticmethod
    def seed_identifier(d: dict) -> str:
        if isinstance(d, EvaluationRunIdentifier):
            d = vars(d)
        return '{s[name]}_{s[challenge]}_{s[library_name]}_{s[model_name]}'.format(s=d)

    @staticmethod
    def run_identifier(d: dict) -> str:
        if isinstance(d, EvaluationRunIdentifier):
            d = vars(d)
        return '{s[name]}_{s[challenge]}_{s[library_name]}_{s[model_name]}_{s[evaluation_type]}'.format(s=d)

    def seed_identifier_dict(self) -> dict:
        identifier = dict(vars(self))
        del identifier['evaluation_type']
        return identifier

    def run_identifier_dict(self) -> dict:
        return dict(vars(self))
