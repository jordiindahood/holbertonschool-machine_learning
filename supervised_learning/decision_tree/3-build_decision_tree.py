#!/usr/bin/env python3
"""
    CLASS FILE FOR DECISION TREE
"""
import numpy as np


class Node:
    """
    NODE CLASS
    """

    def __init__(
        self,
        feature=None,
        threshold=None,
        left_child=None,
        right_child=None,
        is_root=False,
        depth=0,
    ):
        """
        init
        """
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """
        max depth
        """
        if self.is_leaf:
            return self.depth

        left = self.left_child.max_depth_below()
        right = self.right_child.max_depth_below()
        if left > right:
            return left
        return right

    def count_nodes_below(self, only_leaves=False):
        """
        count nodes below
        """
        if only_leaves:
            return self.left_child.count_nodes_below(
                only_leaves
            ) + self.right_child.count_nodes_below(only_leaves)
        else:
            return (
                1
                + self.left_child.count_nodes_below()
                + self.right_child.count_nodes_below()
            )

    def left_child_add_prefix(self, text):
        """
        print left child
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("    |  " + x) + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """
        print right child
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("       " + x) + "\n"
        return new_text

    def __str__(self):
        """
        STR
        """
        if self.is_root:
            Type = "root "
        elif self.is_leaf:
            return f"-> leaf [value={self.value}]"
        else:
            Type = "-> node "
        if self.left_child:
            left_str = self.left_child_add_prefix(str(self.left_child))
        else:
            left_str = ""
        if self.right_child:
            right_str = self.right_child_add_prefix(str(self.right_child))
        else:
            right_str = ""
        return f"{Type}[feature={self.feature}, threshold=\
{self.threshold}]\n{left_str}{right_str}".rstrip()

    def get_leaves_below(self):
        """
        get leaves
        """
        if self.is_leaf:
            return [self]
        leaves = []
        if self.left_child:
            leaves.extend(self.left_child.get_leaves_below())
        if self.right_child:
            leaves.extend(self.right_child.get_leaves_below())
        return leaves


class Leaf(Node):
    """
    LEAF CLASS
    """

    def __init__(self, value, depth=None):
        """
        init
        """
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """
        max depth
        """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """
        count nodes below
        """
        return 1

    def __str__(self):
        """
        STR
        """
        return f"-> leaf [value={self.value}]"

    def get_leaves_below(self):
        """get leave"""
        return [self]


class Decision_Tree:
    """
    DESICION TREE CLASS
    """

    def __init__(
        self, max_depth=10, min_pop=1, seed=0, split_criterion="random", root=None
    ):
        """
        init
        """
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """
        depth
        """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """
        count nodes
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """
        STR REPRESENTATION
        """
        return self.root.__str__() + "\n"

    def get_leaves(self):
        """get leaves"""
        return self.root.get_leaves_below()
