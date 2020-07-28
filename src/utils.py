"""
This is temporary code, it is now under refactoring by the author. It is meant to be a working example for the reviewers
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import networkx as nx
import seaborn as sns

    
def draw_subgraphs(adj_mat, final_partition_nodes, maxw):
    """Displays retrieved subraphs"""

    fig = plt.figure(figsize=(30,10))
    G = nx.from_numpy_array(adj_mat)
    n_graphs = len(final_partition_nodes)

    x = (n_graphs % 2)+2
    y = (n_graphs // 2)+1

    for i,c in enumerate(final_partition_nodes):
        sg = G.subgraph(c)
        if nx.get_edge_attributes(sg,'weight'):
            edges,weights = zip(*nx.get_edge_attributes(sg,'weight').items())
            weights = np.array(weights)
            weights /= (maxw*10)
            sg_density = len(edges) / len(sg.nodes)
            plt.subplot(x,y,i+1)
            plt.title(f"{i} : {sg_density:.2f}")
            nx.draw(sg, node_color='r', with_labels=True, edge_color=weights, width=weights, cmap=plt.cm.jet)
    plt.show()

def plot_spixel_segmentation(img, final_partition_nodes, segments,path="./"):
    """Displays segmentation mask to be overlayed on the image"""
    
    sns.set_palette(FLATUI[:len(final_partition_nodes)])
    color_palette = sns.color_palette()
    rmask = np.zeros(img.shape)
    pred = np.zeros(img.shape[:-1])
    fig = plt.figure(figsize=(30,30))
    plt.subplot(1,2,1)    
    for i,superpixels in enumerate(final_partition_nodes):
        for spixel in superpixels:
            rmask[np.where(segments == spixel)] = color_palette[i]
            pred[np.where(segments == spixel)] = i+1
    plt.imshow(img)

    plt.subplot(1,2,2)
    plt.imshow(img)
    

    plt.imshow(rmask, alpha=0.5,)
    
    if path != "./":
        plt.savefig(path)
    else:
        plt.show()
        sns.palplot(color_palette)
    
    plt.close()
    return pred, rmask    


def get_pred(img, final_partition_nodes,segments):
    """ Get prediction classes """
    pred = np.zeros(img.shape[:-1])
    for i,superpixels in enumerate(final_partition_nodes):
        for spixel in superpixels:
            pred[np.where(segments == spixel)] = i+1
    return pred


def findBestPartition_alt(adj_mat, alpha=1, viz=True):
    """Corrosion algorithm devised
    adj_mat: Adjacency matrix of the graph 
    min_cl: the minimum number of nodes composing a subgraph (default 2)
    alpha: the alpha paramenter of the score function
    viz: parameter to specify if we want to show the score over iterations
    """
    partition_scores = []
    best_npartition = -1
    min_cl=2
    max_score = previous_score =  -np.inf
    adj_mat2 = adj_mat.copy()
    for npartition,min_weight in enumerate(np.unique(adj_mat2)):
        res = np.unique(adj_mat2)
        if res.size >1:
            min_weight = res[1]
        adj_mat2 -= min_weight
        adj_mat2[np.where(adj_mat2 < 0)] = 0
        classes = []
        G = nx.from_numpy_array(adj_mat2)
        score = 0
        for i,c in enumerate(nx.connected_components(G)):
            # Extract the connected component
            sg = G.subgraph(c)
            #print(f"partition:{npartition} cc:{i} n:{sg.nodes}")
            if len(sg.nodes) > min_cl: 
                score += np.array([nx.get_edge_attributes(sg, 'weight')[x] for x in nx.get_edge_attributes(sg, 'weight').keys()]).sum() / len(sg.nodes)
                classes.append(tuple(sg.nodes))


        score += -(i+1)*alpha
        partition_scores.append(score)
        if score > max_score:
            if previous_score != -np.inf and previous_score < score:
                max_score = score
                best_npartition = npartition
            
            best_partition_nodes = classes
            best_partition = adj_mat2.copy()
        previous_score = score
        

    if viz:
        # Visualization
        fig = plt.figure(figsize=(20,8))
        plt.title("Partitions score")
        plt.plot(partition_scores, color='b')
        plt.axvline(x=best_npartition, color='r', linestyle='dashed')
        plt.xticks(range(len(partition_scores)))
        plt.xlabel("Partitions")
        plt.ylabel("Score")
        plt.grid()
        plt.show()
    
    return best_partition_nodes, best_partition

def gkern(kernlen: int=21, nsig: int=3) -> np.ndarray:
    """Returns a 2D Gaussian kernel."""

    x = np.linspace(-nsig, nsig, kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d/kern2d.sum()


def eu_dist(p1:tuple, p2:tuple) -> np.array:
    """Computes euclidead distance"""
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)



FLATUI = ["#FFFF00","#1CE6FF","#FF34FF","#FF4A46","#008941","#006FA6","#A30059","#FFDBE5","#7A4900","#0000A6","#63FFAC","#B79762","#004D43","#8FB0FF","#997D87","#5A0007","#809693","#FEFFE6","#1B4400","#4FC601","#3B5DFF","#4A3B53","#FF2F80","#61615A","#BA0900","#6B7900","#00C2A0","#FFAA92","#FF90C9","#B903AA","#D16100","#DDEFFF","#000035","#7B4F4B","#A1C299","#300018","#0AA6D8","#013349","#00846F","#372101","#FFB500","#C2FFED","#A079BF","#CC0744","#C0B9B2","#C2FF99","#001E09","#00489C","#6F0062","#0CBD66","#EEC3FF","#456D75","#B77B68","#7A87A1","#788D66","#885578","#FAD09F","#FF8A9A","#D157A0","#BEC459","#456648","#0086ED","#886F4C","#34362D","#B4A8BD","#00A6AA","#452C2C","#636375","#A3C8C9","#FF913F","#938A81","#575329","#00FECF","#B05B6F","#8CD0FF","#3B9700","#04F757","#C8A1A1","#1E6E00","#7900D7","#A77500","#6367A9","#A05837","#6B002C","#772600","#D790FF","#9B9700","#549E79","#FFF69F","#201625","#72418F","#BC23FF","#99ADC0","#3A2465","#922329","#5B4534","#FDE8DC","#404E55","#0089A3","#CB7E98","#A4E804","#324E72","#6A3A4C","#83AB58","#001C1E","#D1F7CE","#004B28","#C8D0F6","#A3A489","#806C66","#222800","#BF5650","#E83000","#66796D","#DA007C","#FF1A59","#8ADBB4","#1E0200","#5B4E51","#C895C5","#320033","#FF6832","#66E1D3","#CFCDAC","#D0AC94","#7ED379","#012C58","#7A7BFF","#D68E01","#353339","#78AFA1","#FEB2C6","#75797C","#837393","#943A4D","#B5F4FF","#D2DCD5","#9556BD","#6A714A","#001325","#02525F","#0AA3F7","#E98176","#DBD5DD","#5EBCD1","#3D4F44","#7E6405","#02684E","#962B75","#8D8546","#9695C5","#E773CE","#D86A78","#3E89BE","#CA834E","#518A87","#5B113C","#55813B","#E704C4","#00005F","#A97399","#4B8160","#59738A","#FF5DA7","#F7C9BF","#643127","#513A01","#6B94AA","#51A058","#A45B02","#1D1702","#E20027","#E7AB63","#4C6001","#9C6966","#64547B","#97979E","#006A66","#391406","#F4D749","#0045D2","#006C31","#DDB6D0","#7C6571","#9FB2A4","#00D891","#15A08A","#BC65E9","#FFFFFE","#C6DC99","#203B3C","#671190","#6B3A64","#F5E1FF","#FFA0F2","#CCAA35","#374527","#8BB400","#797868","#C6005A","#3B000A","#C86240","#29607C","#402334","#7D5A44","#CCB87C","#B88183","#AA5199","#B5D6C3","#A38469","#9F94F0","#A74571","#B894A6","#71BB8C","#00B433","#789EC9","#6D80BA","#953F00","#5EFF03","#E4FFFC","#1BE177","#BCB1E5","#76912F","#003109","#0060CD","#D20096","#895563","#29201D","#5B3213","#A76F42","#89412E","#1A3A2A","#494B5A","#A88C85","#F4ABAA","#A3F3AB","#00C6C8","#EA8B66","#958A9F","#BDC9D2","#9FA064","#BE4700","#658188","#83A485","#453C23","#47675D","#3A3F00","#061203","#DFFB71","#868E7E","#98D058","#6C8F7D","#D7BFC2","#3C3E6E","#D83D66","#2F5D9B","#6C5E46","#D25B88","#5B656C","#00B57F","#545C46","#866097","#365D25","#252F99","#00CCFF","#674E60","#FC009C","#92896B"]

def get_cluster_colors(n_clusters, clustering):


    if n_clusters < len(FLATUI):
        sns.set_palette(FLATUI[:n_clusters])
        color_palette = sns.color_palette()
    else:
        FLATUI2 = 100*FLATUI
        sns.set_palette(FLATUI2[:n_clusters])
        color_palette = sns.color_palette()

    cluster_colors = [color_palette[lbl] if lbl >= 0 else (0.5, 0.5, 0.5) for lbl in clustering]
    return cluster_colors
