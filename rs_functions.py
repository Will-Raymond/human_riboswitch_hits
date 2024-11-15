# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 12:52:00 2024

@author: willi
"""


import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
import json

from sklearn import mixture
from pulearn import ElkanotoPuClassifier, BaggingPuClassifier
from sklearn.svm import SVC
from joblib import dump, load
from tqdm import tqdm

#@title functions
import itertools as it
ns_dict = {'m':'a','w':'a','r':'g','y':'t','k':'g','s':'g','w':'a','h':'a','n':'a','x':'a'}

def clean_seq(seq):
    '''
    clean the sequences to lowercase only a, u, g, c
    '''
    seq = seq.lower()
    for key in ns_dict.keys():
        seq = seq.replace(key,ns_dict[key])

    seq = seq.replace('t','u')
    return seq.lower()

def kmer_list(k):
    combos =[x for x in it.product(['a','c','u','g'], repeat=k)]
    kmer = [''.join(y) for y in combos]
    return kmer

def kmer_freq(seq,k=3):
    '''
    calculate the kmer frequences of k size for seq
    '''
    kmer_ind = kmer_list(k)
    kmer_freq_vec = np.zeros((4**k)).astype(int)
    for i in range(len(seq)-k):
        kmer_freq_vec[kmer_ind.index(seq[i:i+k])] += 1

    return kmer_freq_vec

def get_gc(seq):
  return (seq.count('g') + seq.count('c'))/len(seq)


from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib import text as mtext
import numpy as np
import math

#https://stackoverflow.com/questions/19353576/curved-text-rendering-in-matplotlib

class CurvedText(mtext.Text):
    """
    A text object that follows an arbitrary curve.
    """
    def __init__(self, x, y, text, axes, **kwargs):
        super(CurvedText, self).__init__(x[0],y[0],' ', **kwargs)

        axes.add_artist(self)

        ##saving the curve:
        self.__x = x
        self.__y = y
        self.__zorder = self.get_zorder()

        ##creating the text objects
        self.__Characters = []
        for c in text:
            if c == ' ':
                ##make this an invisible 'a':
                t = mtext.Text(0,0,'a')
                t.set_alpha(0.0)
            else:
                t = mtext.Text(0,0,c, **kwargs)

            #resetting unnecessary arguments
            t.set_ha('center')
            t.set_rotation(0)
            t.set_zorder(self.__zorder +1)

            self.__Characters.append((c,t))
            axes.add_artist(t)


    ##overloading some member functions, to assure correct functionality
    ##on update
    def set_zorder(self, zorder):
        super(CurvedText, self).set_zorder(zorder)
        self.__zorder = self.get_zorder()
        for c,t in self.__Characters:
            t.set_zorder(self.__zorder+1)

    def draw(self, renderer, *args, **kwargs):
        """
        Overload of the Text.draw() function. Do not do
        do any drawing, but update the positions and rotation
        angles of self.__Characters.
        """
        self.update_positions(renderer)

    def update_positions(self,renderer):
        """
        Update positions and rotations of the individual text elements.
        """

        #preparations

        ##determining the aspect ratio:
        ##from https://stackoverflow.com/a/42014041/2454357

        ##data limits
        xlim = self.axes.get_xlim()
        ylim = self.axes.get_ylim()
        ## Axis size on figure
        figW, figH = self.axes.get_figure().get_size_inches()
        ## Ratio of display units
        _, _, w, h = self.axes.get_position().bounds
        ##final aspect ratio
        aspect = ((figW * w)/(figH * h))*(ylim[1]-ylim[0])/(xlim[1]-xlim[0])

        #points of the curve in figure coordinates:
        x_fig,y_fig = (
            np.array(l) for l in zip(*self.axes.transData.transform([
            (i,j) for i,j in zip(self.__x,self.__y)
            ]))
        )

        #point distances in figure coordinates
        x_fig_dist = (x_fig[1:]-x_fig[:-1])
        y_fig_dist = (y_fig[1:]-y_fig[:-1])
        r_fig_dist = np.sqrt(x_fig_dist**2+y_fig_dist**2)

        #arc length in figure coordinates
        l_fig = np.insert(np.cumsum(r_fig_dist),0,0)

        #angles in figure coordinates
        rads = np.arctan2((y_fig[1:] - y_fig[:-1]),(x_fig[1:] - x_fig[:-1]))
        degs = np.rad2deg(rads)


        rel_pos = 10
        for c,t in self.__Characters:
            #finding the width of c:
            t.set_rotation(0)
            t.set_va('center')
            bbox1  = t.get_window_extent(renderer=renderer)
            w = bbox1.width
            h = bbox1.height

            #ignore all letters that don't fit:
            if rel_pos+w/2 > l_fig[-1]:
                t.set_alpha(0.0)
                rel_pos += w
                continue

            elif c != ' ':
                t.set_alpha(1.0)

            #finding the two data points between which the horizontal
            #center point of the character will be situated
            #left and right indices:
            il = np.where(rel_pos+w/2 >= l_fig)[0][-1]
            ir = np.where(rel_pos+w/2 <= l_fig)[0][0]

            #if we exactly hit a data point:
            if ir == il:
                ir += 1

            #how much of the letter width was needed to find il:
            used = l_fig[il]-rel_pos
            rel_pos = l_fig[il]

            #relative distance between il and ir where the center
            #of the character will be
            fraction = (w/2-used)/r_fig_dist[il]

            ##setting the character position in data coordinates:
            ##interpolate between the two points:
            x = self.__x[il]+fraction*(self.__x[ir]-self.__x[il])
            y = self.__y[il]+fraction*(self.__y[ir]-self.__y[il])

            #getting the offset when setting correct vertical alignment
            #in data coordinates
            t.set_va(self.get_va())
            bbox2  = t.get_window_extent(renderer=renderer)

            bbox1d = self.axes.transData.inverted().transform(bbox1)
            bbox2d = self.axes.transData.inverted().transform(bbox2)
            dr = np.array(bbox2d[0]-bbox1d[0])

            #the rotation/stretch matrix
            rad = rads[il]
            rot_mat = np.array([
                [math.cos(rad), math.sin(rad)*aspect],
                [-math.sin(rad)/aspect, math.cos(rad)]
            ])

            ##computing the offset vector of the rotated character
            drp = np.dot(dr,rot_mat)

            #setting final position and rotation:
            t.set_position(np.array([x,y])+drp)
            t.set_rotation(degs[il])

            t.set_va('center')
            t.set_ha('center')

            #updating rel_pos to right edge of character
            rel_pos += w-used




#@title dot2circ
import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from typing import TypeVar, Iterable, Tuple
import warnings


class IncompletePairError(Exception):
    """Raised when the given string contains an bracket that is unpaired
    Attributes:
        index -- index of detected unpaired bracket
        message -- exception message
    """

    def __init__(self, index, message):
        self.index = index
        self.message = message



class UnidentifiedCharacterError(Exception):
    """Raised when the given string contains unspecified characters for bracket notations
    Attributes:
        chars -- list of unrecognized characters
        message -- exception message
    """

    def __init__(self, unspecified_character_list, message):
        self.chars = unspecified_character_list
        self.message = message


class ConnectionMatrix():
    def __init__(self):
        self.dot_structure = []
        self.test_seq = '((..(((..((((....))))..))).))......(((((((((..(((((.(((((((.((((((((((..................(((.(((((((...............))))))))))))))))))).).))))))...............).)))))))))...........)))))......'
        self.test_seq_crossing = '((..(((..((((....))))..))).))......(((((((((..(((((.(((((((.((((((((((..................(((.(((((((...............))))))))))))))))))).).))))))...............).)))))))))...........)))))......'


    def convert_to_match_tuples(self, dot_structure_string, str_pair):

        re_special_characters = ['[', ']','?','|','']
        sp1,sp2 =  re.escape(str_pair[0]), re.escape(str_pair[1])

        total_lbrack = len(re.findall(  sp1, dot_structure_string))
        total_rbrack = len(re.findall(  sp2, dot_structure_string))

        if total_lbrack != total_rbrack:
            start_inds = [m.start(0)for m in re.finditer(sp1, dot_structure_string)]
            end_inds = [m.start(0)for m in re.finditer(sp2, dot_structure_string)]
            diff = np.abs(len(end_inds) - len(start_inds))
            if len(start_inds) < len(end_inds):
                unmatched = end_inds[-diff:]
            else:
                unmatched = start_inds[:diff]

            message = 'dotstructure has an incomplete base pair: num("(") and num(")") \
            do not match, double check this dot structure. Indices unmatched: '
            raise IncompletePairError(unmatched , message + str(unmatched) + ' ' + dot_structure_string )

        current_Ls = 0
        current_Rs = 0
        current_Ls_inds = []
        matches = []

        for i in range(len(dot_structure_string)):

            if dot_structure_string[i] ==str_pair[0]:
                current_Ls +=1
                current_Ls_inds = current_Ls_inds + [i,]
            if dot_structure_string[i] ==  str_pair[1]:
                current_Ls -=1
                matches = matches + [(current_Ls_inds[-1], i ), ]
                current_Ls_inds = current_Ls_inds[:-1]

        return matches


    def __clean_rna_sequence(self,string: str,
                             additional_bracket_pairs: Iterable[Iterable] = [[]] ):

        string = string.replace(':','.') #replace colon with .

        total_length =0
        total_length += string.count('.') + string.count('(') + string.count(')')
        test_str = string.replace('.','').replace('(','').replace(')','')

        valid_brackets = []
        for pair in additional_bracket_pairs:
            if len(pair) == 0:
                valid_brackets = valid_brackets + [pair,]

        for pair in valid_brackets:
            if len(pair) > 0:
                total_length += string.count(pair[0]) + string.count(pair[1])
                test_str = test_str.replace(pair[0],'').replace(pair[1],'')

        non_matching_chars = set(test_str)
        custom_chars = [item for sublist in valid_brackets for item in sublist] #specified custom characters

        unaccounted_for_chars = list(set(non_matching_chars) - set(custom_chars))

        if len(unaccounted_for_chars) > 0:
            message = "there are unaccounted for characters in the string, please\
                        add all bracket pairs in the format of a iterable of pairs, example: \
                            custom_brackets = [('a','b')] would allow for dot structure like ...a...b..."
            message = message + ' ' + str(unaccounted_for_chars)
            raise UnidentifiedCharacterError(unaccounted_for_chars, message)
        return string, total_length


    def __generate_seperate_substrings(self,cleaned_string, all_bracket_pairs):
        substrings = []

        for j in range(len(all_bracket_pairs)):
            if len(all_bracket_pairs[j]) > 0:
                substring = cleaned_string
                for i in range(len(all_bracket_pairs)):
                    if i != j:
                        pair = all_bracket_pairs[i]
                        substring = substring.replace(pair[0],'.').replace(pair[1],'.')

                substring  = substring.replace(all_bracket_pairs[j][0],'(').replace(all_bracket_pairs[j][1],')')
                substrings = substrings + [substring,]


        return substrings


    def convert_to_connection_matrix(self,dot_structure_string : str,
                                     custom_brackets : Iterable[Iterable] = [] ) -> np.ndarray:

        cleaned_string,total_length = self.__clean_rna_sequence(dot_structure_string, custom_brackets)
        connection_mat = np.zeros([total_length,]*2)
        all_bracket_pairs = custom_brackets + [['(',')'],]

        valid_brackets = []
        for pair in all_bracket_pairs:
            if len(pair) > 0:
                valid_brackets = valid_brackets + [pair,]


        substrings = self.__generate_seperate_substrings(cleaned_string, valid_brackets)


        for i in range(len(valid_brackets)):

            matches = self.convert_to_match_tuples(substrings[i], ['(',')'])

            for match in matches:
                connection_mat[match[0], match[1]] = 1
                connection_mat[match[1], match[0]] = 1
        return connection_mat.astype(int)


class RNA_CircPlot():
    def __init__(self):
        pass

def pxy(r,angle):
    #convert r, theta to x,y
    return np.array([r*np.cos(angle), r*np.sin(angle)])

def rna_chord_plot( connection_matrix, max_size,ax, sequence=None, **kwargs):

    if max_size < connection_matrix.shape[0]:
        warnings.warn("max_size is less than connection_matrix shape, setting max_size to connection_matrix.shape[0]", Warning)
        max_size = int(connection_matrix.shape[0])

    ax.set_aspect('equal')
    nodes = len(connection_matrix)
    r_circ = 250
    r_text = 270
    center = (50,50)
    offset = 350


    angles = np.linspace(0,2*np.pi-.2,max_size)
    circle_pts = np.array([r_circ*np.cos(angles), r_circ*np.sin(angles)]) + offset

    cset = set(['colors','c','color']).intersection(set(kwargs.keys()))
    if len(cset) > 0:
        c = kwargs[list(cset)[0]]
        kwargs.pop(list(cset)[0])
    else:
        c = ['#57ffcd', '#ff479d', '#ffe869','#ff8c00','#04756f']

    if isinstance(c,str):
        c = [c]

    if sequence != None:
        if len(c) < 4:
            c = c*4
        t = ['a','g','u','c']
        y = list(sequence)
        colors = [c[t.index(x)] for x in y]
    else:
        colors = [c[0],]*nodes
        # c  = ['#ef476f', '#06d6a0','#118ab2','#7400b8','#073b4c', '#118ab2',]
        # c = ['#57ffcd', '#ff479d', '#ffe869','#ff8c00','#04756f']
        # t = ['a','g','u','c']


    circles = [plt.Circle( (circle_pts[0,x],circle_pts[1,x]),5,edgecolor=colors[x],**kwargs  ) for x in range(nodes)]

    k = 0
    for i in range(len(circles)):
        ax.add_patch(circles[i])
        if k%5 == 0:


            ax.annotate(str(k), (pxy(r_text,angles[i]) + offset).tolist(), fontsize=4, ha="center")
        k+=1



    # get the connection indexes, only unique ones
    connection_inds =np.vstack( [np.where(connection_matrix > 0)[0],np.where(connection_matrix > 0)[1]]).T
    connection_inds = connection_inds[np.where(connection_inds[:,0] < connection_inds[:,1]) ]

    bezier_path = np.arange(0, 1.02, 0.02)
    center_node = [offset,offset]


    kwargs.pop('facecolor')
    for i in range(len(connection_inds)):

        loc1 = circle_pts[:,connection_inds[i][0] ]
        loc2 = circle_pts[:,connection_inds[i][1] ]

        x1,y1 = loc1
        x2,y2 = loc2
        xb,yb = center_node
        #quadratic bzeir curve through loc1 and loc2 with bounding point as the center of the circle
        x = (1-bezier_path)**2 * x1 + 2*(1-bezier_path)*bezier_path*xb + bezier_path**2*x2
        y = (1-bezier_path)**2*y1 + 2*(1-bezier_path)*bezier_path*yb + bezier_path**2*y2

        ax.plot(x, y,color = colors[connection_inds[i][0]],**kwargs)




    ax.set_xlim([30,750])
    ax.set_ylim([30,670])
    ax.axis('off')
    if sequence != None:

        legend_elements = [ Line2D([0], [0], marker='o', color=c[x], label=t[x],markerfacecolor='w',lw=0, markersize=3) for x in range(0,4)]
        tts = ['au','gc','ua','cg']
        legend_elements2 = [ Line2D([0], [0], color=c[x], label=tts[x],lw=1, markersize=3) for x in range(0,4)]

        ax.legend(handles=legend_elements+legend_elements2, loc='best',fontsize=7)



#seq = 'gcuuuuguccugcgcgcgcagauuaacgcaaacccggaagcggaucggguggaguguaggucauaucgccgcguugacuagauaacgaaggacaaugcauauauacaaaauugcuguauugccuggugauaugcagcuguaccugugacaucauaauugcacccuccgacaugauaucucuuccaaaugc'
#test_seq = '((..(((..((((....))))..))).))......(((((((((..(((((.(((((((.((((((((((..................(((.(((((((...............))))))))))))))))))).).))))))...............).)))))))))...........)))))......'

# seq = 'gcuuuuguccugcgcgcgcagauuaacgcaaacccggaagcggaucggguggaguguaggucauaucgccgcguugacuagauaacgaaggacaaugcauauauacaaaauugcuguauugccuggugauaugcagaacuguaccugugacaucauaauugcacccuccgacaugauaucucuuccaaaugc'
# test_seq = '((..(((..((((....))))..))).))......(((((((((..(((((.(((((((.((((((((((..........)........(((.(((((((...............))))))))))))))))))).).))))))...............).))))))))...........)))))......'


# test_bracket = '::aa::::::bb::::::(((:::[[[[[[)))::]]]]]]:::'

# test_bracket = '::aa:::11:::bb:::22:::(((:::[[[[[[)))::]]]]]]:::'
#test_bracket = '::aa:::11:::bb:::22:::(((:::[[[[[[[)))::]]]]]]:::'
#c = ConnectionMatrix()
##cmat = c.convert_to_connection_matrix(test_seq, custom_brackets = [[]])
#plt.figure(dpi=300)
#rna_chord_plot(cmat,50,sequence=seq, facecolor='w',lw=1,)


# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 11:33:41 2022

@author: willi
"""

import numpy as np
import re
from typing import TypeVar, Iterable, Tuple
import warnings

class IncompletePairError(Exception):
    """Raised when the given string contains an bracket that is unpaired

    Attributes:
        index -- index of detected unpaired bracket
        message -- exception message
    """

    def __init__(self, index, message):
        self.index = index
        self.message = message



class UnidentifiedCharacterError(Exception):
    """Raised when the given string contains unspecified characters for bracket notations

    Attributes:
        chars -- list of unrecognized characters
        message -- exception message
    """

    def __init__(self, unspecified_character_list, message):
        self.chars = unspecified_character_list
        self.message = message

# custom bear like encoder to count dot structure features
class BEARencoder():
    def __init__(self):
        self.stem_dict = ['a','b','c','d','e','f','g','h','i','j','-','|','A','B','C','D','w','x','y','z']
        self.loop_dict = ['m','n','o','p','q','r','s','t','u','v','E','F','G','H','T','U','V','W','X','Y','Z','ž','¡','§','©','®','±','¶','·','»']
        #self.intloop_dict = ['w','x','y','z', '@','*','>','<','?','~','_','I','J']
        self.bulge_l = '['
        self.bulge_r = ']'

        self.intloop_dict_r = ['+','2','3','4','5','6','7','8','9','0','=','+','-',';','K','L','M','@','*','>',]
        self.intloop_dict_l = ['^','!','"','#','$','%','&',"'",'(',')','}','{','N','O','P','R','S','?','~','_',]

        self.test_str = '((((.((((.....)))..).))))'

        self.loop_regex = re.compile('\(\.*\)')
        self.bulge_l_regex = re.compile(r'(\(\.{1}(?=\())')
        self.bulge_r_regex = re.compile(r'(\)\.{1}(?=\)))')

        self.internal_l_regex = re.compile(r'(\(\.{2,}(?=\())')
        self.internal_r_regex = re.compile(r'(\)\.{2,}(?=\)))')


        self.stem_limit = len(self.stem_dict)
        self.loop_limit = len(self.loop_dict)
        self.ilr_limit = len(self.intloop_dict_r)
        self.ill_limit = len(self.intloop_dict_l)

    def convert_to_match_tuples(self, dot_structure_string, str_pair):

        re_special_characters = ['[', ']','?','|','']
        sp1,sp2 =  re.escape(str_pair[0]), re.escape(str_pair[1])

        total_lbrack = len(re.findall(  sp1, dot_structure_string))
        total_rbrack = len(re.findall(  sp2, dot_structure_string))

        if total_lbrack != total_rbrack:
            start_inds = [m.start(0)for m in re.finditer(sp1, dot_structure_string)]
            end_inds = [m.start(0)for m in re.finditer(sp2, dot_structure_string)]
            diff = np.abs(len(end_inds) - len(start_inds))
            if len(start_inds) < len(end_inds):
                unmatched = end_inds[-diff:]
            else:
                unmatched = start_inds[:diff]

            message = 'dotstructure has an incomplete base pair: num("(") and num(")") \
            do not match, double check this dot structure. Indices unmatched: '
            raise IncompletePairError(unmatched , message + str(unmatched) + ' ' + dot_structure_string )

        current_Ls = 0
        current_Rs = 0
        current_Ls_inds = []
        matches = []

        for i in range(len(dot_structure_string)):

            if dot_structure_string[i] ==str_pair[0]:
                current_Ls +=1
                current_Ls_inds = current_Ls_inds + [i,]
            if dot_structure_string[i] ==  str_pair[1]:
                current_Ls -=1
                matches = matches + [(current_Ls_inds[-1], i ), ]
                current_Ls_inds = current_Ls_inds[:-1]

        return matches


    def get_non_branching_elements(self, dot_structure, str_pair=['(',')']):
        x=1


    def group_bp_matches(self, matches):
        mL, mR = matches[0]
        count = 1

        label = ['-']
        for i in range(1, len(matches)):
            mL_next, mR_next = matches[i]
            if abs((mL - mL_next)) == 1 and abs((mR-mR_next)) == 1: #matches are still linked
                count+=1
                label.append('-')
            else:
                if count > self.stem_limit:
                    count = self.stem_limit
                newchar = self.stem_dict[count-1]
                for j in range(len(label)):
                    if label[j] == '-':
                        label[j] = newchar
                count = 1
                label.append('-')
            mL, mR = mL_next, mR_next

        if count > 16:
            count = 16
        newchar = self.stem_dict[count-1]
        for j in range(len(label)):
            if label[j] == '-':
                label[j] = newchar


        return label


    def is_pair_multibranched(self,substring):
        loop_layer = 0
        layer_counts = [0,]
        for i in range( len(substring)):
            if substring[i] == '(':
                loop_layer+=1
            if substring[i] == ')':
                loop_layer-=1

            layer_counts.append(loop_layer)
        layer_counts = np.array(layer_counts)
        diff = np.array(layer_counts)[1:] - np.array(layer_counts)[:-1]
        last_positive = np.where(diff > 0)[0][-1]
        return np.any(diff[:last_positive] < 0)

    def get_loops(self, dot_structure):
        matches = [(m.start(0), m.end(0)) for m in self.loop_regex.finditer(dot_structure)]
        labels = []
        matches_sub_one = []
        for match in matches:
            loop_len = len(dot_structure[match[0]+1:match[1]-1])
            if loop_len > self.loop_limit+1:
                loop_len = self.loop_limit+1
            labels.append(self.loop_dict[loop_len-3] )
            matches_sub_one.append((match[0]+1,match[1]-1))

        return matches_sub_one, labels


    def get_internal_loops(self, dot_structure):
        l_internal_loops = [(m.start(0), m.end(0)) for m in self.internal_l_regex.finditer(dot_structure)]
        r_internal_loops = [(m.start(0), m.end(0)) for m in self.internal_r_regex.finditer(dot_structure)]

        l_labels = []
        r_labels = []
        l_matches = []
        r_matches = []

        for match in l_internal_loops:
            loop_len = len(dot_structure[match[0]+1:match[1]])
            if loop_len > self.ill_limit:
                loop_len = self.ill_limit
            l_labels.append(self.intloop_dict_l[loop_len-1] )
            l_matches.append((match[0]+1,match[1]))

        for match in r_internal_loops:
            loop_len = len(dot_structure[match[0]+1:match[1]])
            if loop_len > self.ilr_limit:
                loop_len = self.ilr_limit
            r_labels.append(self.intloop_dict_r[loop_len-1] )
            r_matches.append((match[0]+1,match[1]))
        return l_matches, l_labels, r_matches, r_labels

    def get_unmatched(self,dot_structure):
        x=1


    def condense_unbranched_loops(self, stems):
        cond_stem = []
        for stem in stems:
            l,r = stem
            if not any([i[0] < l and i[1] > r for i in stems]):
                cond_stem.append(stem)
        return cond_stem


    def get_bulges(self,dot_structure):
        l_bulges = [(m.start(0),m.end(0)) for m in self.bulge_l_regex.finditer(dot_structure)]
        r_bulges = [(m.start(0),m.end(0)) for m in self.bulge_r_regex.finditer(dot_structure)]
        l_labels = []
        r_labels = []
        l_matches = []
        r_matches = []
        for match in l_bulges:
            loop_len = len(dot_structure[match[0]+1:match[1]])
            l_labels.append('[')
            l_matches.append((match[0]+1,match[1]))
        for match in r_bulges:
            loop_len = len(dot_structure[match[0]+1:match[1]])
            r_labels.append(']')
            r_matches.append((match[0]+1,match[1]))

        return l_matches, l_labels, r_matches, r_labels

    def encode(self, dot_structure):

        stems = self.convert_to_match_tuples(dot_structure, ['(',')'])
        if len(stems) ==  0:
            return dot_structure.replace('.',':')
        stem_labels = self.group_bp_matches(stems)



        branched_stems = [stem for stem in stems if self.is_pair_multibranched(dot_structure[stem[0]:stem[1]+1])]
        unbranched_stems = [stem for stem in stems if not self.is_pair_multibranched(dot_structure[stem[0]:stem[1]+1])]

        #branched_stem_labels = [stem_labels[i] for i in range(len(stems)) if self.is_pair_multibranched(stems[i])]
        #unbranched_stem_labels = [stem_labels[i] for i in range(len(stems)) if not self.is_pair_multibranched(stems[i])]

        condensed_unbranched_stems = self.condense_unbranched_loops(unbranched_stems)


        loops, loop_labels = self.get_loops(dot_structure)

        l_bulges, l_labels, r_bulges, r_labels = self.get_bulges(dot_structure)

        ill_loops, ill_labels, ilr_loops, ilr_labels = ([],[],[],[])


        for unbranched_stem in condensed_unbranched_stems:
            start = unbranched_stem[0]
            il, ill, ilr, ilrl = self.get_internal_loops(dot_structure[unbranched_stem[0]:unbranched_stem[1]+1])

            il = [(i[0]+start, i[1]+start) for i in il]

            ilr = [(i[0]+start, i[1]+start) for i in ilr]

            ill_loops = ill_loops + il
            ill_labels = ill_labels + ill
            ilr_loops = ilr_loops + ilr
            ilr_labels = ilr_labels + ilrl


        encoded_string = list(dot_structure)

        for i in range(len(stems)): #convert stems
            encoded_string[stems[i][0]] = stem_labels[i]
            encoded_string[stems[i][1]] = stem_labels[i]

        for i in range(len(ill_loops)): #convert left internal loops
            for j in range(*ill_loops[i]):
                encoded_string[j] = ill_labels[i]


        for i in range(len(ilr_loops)): #convert right internal loops
            for j in range(*ilr_loops[i]):
                encoded_string[j] = ilr_labels[i]

        for i in range(len(loops)): #convert stem end loops
            for j in range(*loops[i]):
                encoded_string[j] = loop_labels[i]

        for i in range(len(l_bulges)): #convert left bulges
            for j in range(*l_bulges[i]):
                encoded_string[j] = l_labels[i]

        for i in range(len(r_bulges)): #convert right bulges
            for j in range(*r_bulges[i]):
                encoded_string[j] = r_labels[i]

        return ''.join(encoded_string).replace('.',':')



    def annoated_feature_vector(self, dot_structure, encode_stems_per_bp=False):

        stems = self.convert_to_match_tuples(dot_structure, ['(',')'])
        if len(stems) ==  0:
            return np.array([0,0,  0,0, 0, 0, 0, 1])

        stem_labels = self.group_bp_matches(stems)


        branched_stems = [stem for stem in stems if self.is_pair_multibranched(dot_structure[stem[0]:stem[1]+1])]
        unbranched_stems = [stem for stem in stems if not self.is_pair_multibranched(dot_structure[stem[0]:stem[1]+1])]

        #branched_stem_labels = [stem_labels[i] for i in range(len(stems)) if self.is_pair_multibranched(stems[i])]
        #unbranched_stem_labels = [stem_labels[i] for i in range(len(stems)) if not self.is_pair_multibranched(stems[i])]

        condensed_unbranched_stems = self.condense_unbranched_loops(unbranched_stems)


        loops, loop_labels = self.get_loops(dot_structure)

        l_bulges, l_labels, r_bulges, r_labels = self.get_bulges(dot_structure)

        ill_loops, ill_labels, ilr_loops, ilr_labels = ([],[],[],[])


        for unbranched_stem in condensed_unbranched_stems:
            start = unbranched_stem[0]
            il, ill, ilr, ilrl = self.get_internal_loops(dot_structure[unbranched_stem[0]:unbranched_stem[1]+1])

            il = [(i[0]+start, i[1]+start) for i in il]

            ilr = [(i[0]+start, i[1]+start) for i in ilr]

            ill_loops = ill_loops + il
            ill_labels = ill_labels + ill
            ilr_loops = ilr_loops + ilr
            ilr_labels = ilr_labels + ilrl


        encoded_string = list(dot_structure)



        for i in range(len(stems)): #convert stems
            encoded_string[stems[i][0]] = stem_labels[i]
            encoded_string[stems[i][1]] = stem_labels[i]

        for i in range(len(ill_loops)): #convert left internal loops
            for j in range(*ill_loops[i]):
                encoded_string[j] = ill_labels[i]


        for i in range(len(ilr_loops)): #convert right internal loops
            for j in range(*ilr_loops[i]):
                encoded_string[j] = ilr_labels[i]

        for i in range(len(loops)): #convert stem end loops
            for j in range(*loops[i]):
                encoded_string[j] = loop_labels[i]

        for i in range(len(l_bulges)): #convert left bulges
            for j in range(*l_bulges[i]):
                encoded_string[j] = l_labels[i]

        for i in range(len(r_bulges)): #convert right bulges
            for j in range(*r_bulges[i]):
                encoded_string[j] = r_labels[i]


        def ranges(nums):
            nums = sorted(set(nums))
            gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s+1 < e]
            edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
            return list(zip(edges, edges))


        final_str = ''.join(encoded_string).replace('.',':')

        s_array = np.array(unbranched_stems)
        used_stems = []
        combined_unbranched_stems = 0
        for i in range(len(unbranched_stems)):
            if i not in used_stems:

                #all indices that are consecutive up / down in one direction with the
                # given base pair, example (46, 52) - (45,53) = (+1,-1) = sum of 0
                matches = np.where(np.sum(s_array-s_array[i],axis=1) == 0)[0]

                #however this doesnt catch pairs that are consecutive with a gap, so
                #extremely convoluted way to find indices that count up consecultively:

                #eg this will catch (46, 52) - (45,53), (44,54) but not (26, 72)
                #                               (-1,+1),(-2,+2)    X    (-20,+20)
                hits = s_array[matches][:,0]
                rs = ranges(hits)
                subgroups = [s_array[i][0] in range(x[0],x[1]+1) for x in rs]
                st,en = rs[np.where(subgroups)[0][0]]
                final_matches = list(range(st,en+1 ))
                final_inds = []
                for m in matches:
                    if s_array[m,0] in final_matches:
                        final_inds.append(m)


                combined_unbranched_stems += 1
                used_stems = used_stems + final_inds

        s_array = np.array(branched_stems)
        used_stems = []
        combined_branched_stems = 0
        for i in range(len(branched_stems)):
            if i not in used_stems:
                #all indices that are consecutive up / down in one direction with the
                # given base pair, example (46, 52) - (45,53) = (+1,-1) = sum of 0
                matches = np.where(np.sum(s_array-s_array[i],axis=1) == 0)[0]

                #however this doesnt catch pairs that are consecutive with a gap, so
                #extremely convoluted way to find indices that count up consecultively:

                #eg this will catch (46, 52) - (45,53), (44,54) but not (26, 72)
                #                               (-1,+1),(-2,+2)    X    (-20,+20)
                hits = s_array[matches][:,0]
                rs = ranges(hits)
                subgroups = [s_array[i][0] in range(x[0],x[1]+1) for x in rs]
                st,en = rs[np.where(subgroups)[0][0]]
                final_matches = list(range(st,en+1 ))
                final_inds = []
                for m in matches:
                    if s_array[m,0] in final_matches:
                        final_inds.append(m)


                combined_branched_stems += 1
                used_stems = used_stems + final_inds


        if encode_stems_per_bp:

            return np.array([combined_unbranched_stems, combined_branched_stems,   len(ill_loops), len(ilr_loops), len(loops), len(l_bulges), len(r_bulges), final_str.count(':')/len(final_str)])
        else:
            return np.array([len(unbranched_stems), len(branched_stems),   len(ill_loops), len(ilr_loops), len(loops), len(l_bulges), len(r_bulges), final_str.count(':')/len(final_str)])

