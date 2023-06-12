import numpy as np
import matplotlib.pyplot as plt

import ahocorasick


# NOT FOR USE WITHIN DATALOADER THREADS (sic!)

class SeqEncoder:
    def __init__(self, alphabet=("A", "G", "C", "T")):
        self.alphabet = np.array(alphabet)
        self.code = {n: i for n, i in zip(self.alphabet, range(len(self.alphabet)))}
        self.code_r = {v: k for k, v in self.code.items()}
    
    def encode(self, sequence_str):
        sequence_labels = [self.code[i] for i in sequence_str]
        return np.eye(self.alphabet.__len__())[sequence_labels].T
    
    def decode(self, sequence_array):
        sequence_labels = sequence_array.argmax(axis=0)
        sequence_string = "".join(self.alphabet[sequence_labels])
        return sequence_string


class MotifsAutomaton:
    def __init__(self, motifs: list, check=True, alphabet=("A", "G", "C", "T")):
        self.enc = SeqEncoder(alphabet=alphabet)
        if check:
            motif_dtypes = tuple(set([type(m) for m in motifs]))
            if len(motif_dtypes) > 1:
                raise TypeError("Passed motifs must be of the same type")
            if motif_dtypes[0].__module__ == np.__name__:
                motifs = [self.enc.decode(m) for m in motifs]
            elif motif_dtypes[0] is str:
                pass
            else:
                raise TypeError(f"Cannot parse motifs of type {motif_dtypes[0]}")
        self.motifs = motifs
        lengths = set([len(m) for m in motifs])
        self.motif_length = None if len(lengths) != 1 else next(iter(lengths))
        self.enc_motifs = [self.enc.encode(m) for m in motifs]
        self.make_ahocorasick()
    
    def make_ahocorasick(self):
        self.aho = ahocorasick.Automaton()
        for idx, m in enumerate(self.motifs):
            self.aho.add_word(m, (idx, m))
        self.aho.make_automaton()
    
    def findings_exact(self, sequence_text_str):
        # O(N) algorithm
        findings = np.zeros(len(sequence_text_str))
        for end_index, (_, motif) in self.aho.iter(sequence_text_str):
            start_index = end_index - len(motif) + 1
            findings[start_index:start_index + len(motif)] += 1
        return findings
    
    def similarity_inexact(self, sequence_text_array):
        # O(M**2 * N * |A|) algorithm
        similarities = list()
        for m_array in self.enc_motifs:
            motif_length = m_array.shape[-1]
            sim = self.similarity_arrays(sequence_text_array, m_array)
            if self.motif_length is None:
                sim = self.similarity_array_highlightmax(sim, motif_length)
            similarities.append(sim)
        sim_stacked = np.stack(similarities).max(axis=0)
        if self.motif_length is not None:
            sim_stacked = self.similarity_array_highlightmax(sim_stacked, self.motif_length)
        return sim_stacked
    
    def similarity_arrays(self, sequence_text_array, sequence_motif_array):
        seq_text_flat = sequence_text_array.ravel(order="F")
        seq_motif_flat = sequence_motif_array.ravel(order="F")
        # convolve without flipping (cross-correlate):
        similarity = np.correlate(seq_text_flat, seq_motif_flat)
        # remove incorrect shifts (due to 2D structure):
        similarity = similarity[::4]
        # pad with zeros to save shape:
        pad_length = sequence_text_array.shape[1] - similarity.shape[0]
        sim_padded = np.concatenate((similarity,
                                     np.zeros(pad_length),
                                    ))
        sim = sim_padded / sim_padded.max()
        return sim
    
    def similarity_array_highlightmax(self, sim, highlight_length):
        sims = [sim]
        rolls = highlight_length
        for i in range(1, rolls):
            sims.append(np.roll(sim, i))
        sim_max = np.stack(sims).max(axis=0)
        return sim_max


class MotifsTracer(MotifsAutomaton):
    def __init__(self, exact=True, inexact=True, **kwargs):
        super().__init__(**kwargs)
        self.exact = exact
        self.inexact = inexact
    
    def get_vis_arrays(self, sequence_text):
        if isinstance(sequence_text, str):
            sequence_text_str = sequence_text
            sequence_text_array = self.enc.encode(sequence_text)
        else:
            sequence_text_array = sequence_text
            sequence_text_str = self.enc.decode(sequence_text)
        out_arr = sequence_text_array
        if self.exact:
            exact_arr = self.findings_exact(sequence_text_str)
            out_arr = self.add_channel(out_arr, exact_arr, intensify_func=lambda x: np.log2(x + 1))
        if self.inexact:
            inexact_arr = self.similarity_inexact(sequence_text_array)
            out_arr = self.add_channel(out_arr, inexact_arr, intensify_func=lambda x: 1.2 ** x)
        mask_all = np.ones_like(out_arr, dtype=bool)
        if self.exact and self.inexact:
            mask_all[-2:,:] = False
            mask_exact = np.zeros_like(mask_all, dtype=bool)
            mask_exact[-2] = True
            mask_inexact = np.zeros_like(mask_all, dtype=bool)
            mask_inexact[-1] = True
        elif self.exact and not self.inexact:
            mask_all[-1:,:] = False
            mask_exact = np.zeros_like(mask_all, dtype=bool)
            mask_exact[-1] = True
            mask_inexact = np.zeros_like(mask_all, dtype=bool)
        elif not self.exact and self.inexact:
            mask_all[-1:,:] = False
            mask_exact = np.zeros_like(mask_all, dtype=bool)
            mask_inexact = np.zeros_like(mask_all, dtype=bool)
            mask_inexact[-1] = True
        else:
            mask_exact = np.zeros_like(mask_all, dtype=bool)
            mask_inexact = np.zeros_like(mask_all, dtype=bool)
        return out_arr, mask_all, mask_exact, mask_inexact
    
    @staticmethod
    def add_channel(src_array, added_array, intensify_func=None):
        if intensify_func is not None:
            added_array = intensify_func(added_array)
        added_array = added_array[np.newaxis,:]
        extended_src_array = np.concatenate((src_array, added_array), axis=0)
        return extended_src_array
    
    def visualize_similarity(
        self,
        sequence_text: str,
        ax=None,
        cmap1="Greys_r",
        cmap2="Wistia",
        cmap3="Wistia",
        **kwargs
    ):
        kwargs.pop("cmap", None)
        
        out_arr, mask_all, mask_exact, mask_inexact = self.get_vis_arrays(sequence_text)
        # mask for visualization
        if ax is None:
            _, ax = plt.subplots(1, 1)
        labels = list(self.enc.alphabet)
        ax.imshow(np.ma.masked_array(out_arr, ~mask_all), cmap=cmap1, **kwargs)
        if self.exact:
            ax.imshow(np.ma.masked_array(out_arr, ~mask_exact), cmap=cmap2, **kwargs)
            labels.append("Exact")
        if self.inexact:
            ax.imshow(np.ma.masked_array(out_arr, ~mask_inexact), cmap=cmap3, **kwargs)
            labels.append("~Sim")
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels)
        
        # set spines invisible
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        # ax.spines['bottom'].set_visible(False)
        # ax.spines['left'].set_visible(False)
        return None

if __name__ == "__main__":
    seq1 = "AAAAGTAGCTACGTAGCTAGTCGATCGTAGCTGCATCGTAGTTTATCGTAGCGCGATCGGGA"
    seq2 = "AAAAGTAGGTACGAAGCTAGTCGATCATTGCTGCATCGTAGTTTAACGTAGCGCGATCGGGA"
    motifs = ["GTAGCTA", "CGTAGCTGCAT", "TGCATCGTAGT", "TTTTT"]
    
    fig, ax = plt.subplots(2, 1, figsize=(20, 5))
    MotifsTracer(motifs=motifs).visualize_similarity(seq1, ax=ax[0])
    MotifsTracer(motifs=motifs).visualize_similarity(seq2, ax=ax[1])
    plt.show()