class NodeType(object):
    def __init__(self, name: str, perception_range: int) -> None:
        self.name = name
        self.perception_range = perception_range

    def __str__(self):
        return self.name

pedestrian = NodeType(name='pedestrian', perception_range=100)
