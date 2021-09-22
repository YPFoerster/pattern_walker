import unittest
import pattern_walker.utils as utils
import networkx as nx

c=3 #offspring number
h=2



class utilsTestcase(unittest.TestCase):

    def test_size(self):
        H,root=utils.balanced_ditree(c,h)
        self.assertEqual( len(H),(c**(h+1)-1)/(c-1) )

    def test_leaves(self):
        H,root=utils.balanced_ditree(c,h)
        leaves=utils.leaves(H)
        self.assertEqual( len(leaves),c**h )


    def test_parts(self):
        H,root=utils.balanced_ditree(c,h)
        parts=nx.neighbors(H,root)
        parts=list(parts)
        self.assertEqual(len(parts),c)
        leaves=utils.leaves(H)
        part_leaves=[[node for node in  H.successors(part) if node in leaves] for part in parts]
        self.assertEqual( len(part_leaves[0]),len(part_leaves[-1]) )
        self.assertEqual( len(part_leaves[0]), c**(h-1) )



if __name__ == '__main__':
    unittest.main()
