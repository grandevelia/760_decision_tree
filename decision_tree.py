#In[1]
import pandas as pd
import numpy as np
from treelib import Tree
import math

import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error
    
from scipy.interpolate import lagrange
from numpy.polynomial.polynomial import Polynomial
import random
pd.options.mode.chained_assignment = None

class ZeroSplitEntropyException(Exception):
    def __init__(self, message):            
        # Call the base class constructor with the parameters it needs
        super().__init__(message)
    

def determine_candidate_splits(data):
    candidate_splits = {}
    dims = ["x", "y"]
    for dim in dims:
        dim_splits = []
        sorted_data = data.sort_values(dim).reset_index(drop=True)
        for i in range(sorted_data.shape[0] - 1):
            if sorted_data.loc[i, "label"] != sorted_data.loc[i+1, "label"]: #If labels don't match, this is a split for this dimension
                dim_splits += [sorted_data.loc[i+1, dim]]
        candidate_splits[dim] = dim_splits
    return candidate_splits


def entropy_value(ratio):
    if ratio == 0 or ratio == 1: return 0
    return ratio * math.log(ratio, 2)


def get_data_entropy(data):
    label_ratios = data['label'].value_counts()/data.shape[0]
    label_entropies = label_ratios.apply(entropy_value)
    return -label_entropies.sum()


def subset_data(data, dim, split_floor, gt=True):
    dim_data = data[dim]
    if gt:
        split_data = data[dim_data >= split_floor]
    else:
        split_data = data[dim_data < split_floor]
    return split_data


def conditional_entropy(data, dim, split_floor):
    total_entropy = 0
    right_split_data = subset_data(data, dim, split_floor)
    right_split_ratio = right_split_data.shape[0]/data.shape[0]
    left_split_data = subset_data(data, dim, split_floor, gt=False)
    left_split_ratio = left_split_data.shape[0]/data.shape[0]
    total_entropy = (
        right_split_ratio * get_data_entropy(right_split_data) +
        left_split_ratio * get_data_entropy(left_split_data)
    )
    return total_entropy


def information_gain(data, dim, split_floor):
    data_entropy = get_data_entropy(data)
    return data_entropy - conditional_entropy(data, dim, split_floor)
    
    
def get_split_entropy(data, dim, split_floor):
    left_ratio = data[dim][data[dim] < split_floor].shape[0]/data.shape[0]
    right_ratio = data[dim][data[dim] >= split_floor].shape[0]/data.shape[0]
    return -(entropy_value(left_ratio) + entropy_value(right_ratio))
    
def gain_ratio(data, dim, split_floor):
    split_entropy = get_split_entropy(data, dim, split_floor)
    if split_entropy == 0:
        raise ZeroSplitEntropyException(f"Info gain was: {information_gain(data, dim, split_floor)} while splitting at {dim} >= {split_floor}")
    return information_gain(data, dim, split_floor)/split_entropy


def stopping_criteria(candidate_splits, data):
    '''
    The stopping criteria (for making a node into a leaf) are that the node is empty, 
    or all splits have zero gain ratio (if the entropy of the split is non-zero),
    or the entropy of any candidates split is zero
    '''
    stop = False
    if data.shape[0] <= 1: return True, []
    gain_ratios = {}
    for dim, dim_splits in candidate_splits.items():
        gain_ratios[dim] = []
        for split_floor in dim_splits:
            try:
                gain_ratios[dim] += [gain_ratio(data, dim, split_floor)]
            except ZeroSplitEntropyException as e:
                print(f"A split had zero entropy.", e)
                #data.to_csv("blabla.csv", index=False)
                #raise Exception (f"FIRST, {dim}, {split_floor}")
                #gain_ratios[dim] += [0]
                #stop = True
                return True, []
    if np.array(
        [np.array(dim_ratios).sum() 
        for dim_ratios in gain_ratios.values()]
    ).sum() == 0: stop = True
    return stop, gain_ratios


def find_best_split(gain_ratios):
    best_dim = ""
    best_index = 0
    best_ratio = -1
    for dim, ratios in gain_ratios.items():
        for i, ratio in enumerate(ratios):
            if ratio > best_ratio:
                best_dim = dim
                best_index = i
                best_ratio = ratio
    return best_dim, best_index


def make_leaf_node(tree, root_id, data, n_nodes):
    ratio = data['label'].sum()/data.shape[0]
    leaf_value = 1
    if ratio < 0.5:
        leaf_value = 0
    tree.create_node(f"{leaf_value}", n_nodes, parent=root_id, data={"pred": leaf_value})
    n_nodes += 1
    return n_nodes


def make_internal_node(tree, root_id, best_split_dim, split_floor, n_nodes):
    tree.create_node(f"{best_split_dim} >= {split_floor}", n_nodes, parent=root_id, data={best_split_dim: split_floor})
    return n_nodes, n_nodes + 1


def make_subtree(tree, root_id, data, n_nodes, print_root_cuts=False):
    #print("Data shape:", data.shape)
    candidate_splits = determine_candidate_splits(data)
    stop, gain_ratios = stopping_criteria(candidate_splits, data)
    if print_root_cuts and root_id == 1:
        print(f"Root candidate cuts: \n {candidate_splits} \n gain ratios: \n {gain_ratios}")
    if stop:
        n_nodes = make_leaf_node(tree, root_id, data, n_nodes)
    else:
        best_split_dim, best_index = find_best_split(gain_ratios)
        split_floor = candidate_splits[best_split_dim][best_index]
        #print(best_split_dim, best_index, split_floor, n_nodes)
        node_id, n_nodes = make_internal_node(tree, root_id, best_split_dim, split_floor, n_nodes)
        #In split
        right_split_data = subset_data(data, best_split_dim, split_floor)
        tree, n_nodes = make_subtree(tree, node_id, right_split_data, n_nodes)
        #Not in split
        left_split_data = subset_data(data, best_split_dim, split_floor, gt=False)
        tree, n_nodes = make_subtree(tree, node_id, left_split_data, n_nodes)
    return tree, n_nodes


def run_dataset(name, df=False, plot=True, **kwargs):
    if df is False:
        fn = f"{name}.txt"
        df = pd.read_csv(f"data/{fn}", sep=" ", names=["x", "y", "label"], comment="#")
    if plot:
        plt.figure(figsize=(20, 20), dpi=100)
        plt.scatter(df["x"], df["y"], c=df["label"], s=200)
        plt.savefig(f"{name}.png", dpi=100)
        plt.close()
    n_nodes = 1
    tree = Tree()
    tree.create_node("root", n_nodes)
    tree, n_nodes = make_subtree(tree, n_nodes, df, n_nodes + 1, **kwargs)
    return tree, df


# %%
tree, df = run_dataset("greedy")

#  %%
tree, df = run_dataset("Druns", print_root_cuts=True)


# %%

def tree_predict(tree, root, data):
    if not tree[root].is_leaf():
        rule_dict = tree[root].data
        rule_dim = list(rule_dict.keys())[0]
        rule_boundary = rule_dict[rule_dim]
        children = tree.is_branch(root)
        for i, child in enumerate(children):
            if i == 0:
                mask = data[rule_dim] >= rule_boundary
            else:
                mask = data[rule_dim] < rule_boundary
            data_subset = data[mask]
            data_subset = tree_predict(tree, child, data_subset)
            data = pd.concat([data[~mask], data_subset])
    else:
        data.loc[:, 'pred'] = tree[root].data['pred']
    return data

# %%
        
def visualize_decision_boundary(tree, data, name):
    feature_1, feature_2 = np.meshgrid(
        np.linspace(data.loc[:, "x"].min(), data.loc[:, "x"].max()),
        np.linspace(data.loc[:, "y"].min(), data.loc[:, "y"].max())
    )
    grid = pd.DataFrame(np.vstack(
        [feature_1.ravel(), feature_2.ravel()]).T
    ).rename(columns={0: "x", 1: "y"})
    grid["pred"] = -1
    preds = tree_predict(tree, 2, grid)['pred'].sort_index()
    y_pred = np.reshape(np.array(preds), feature_1.shape)
    display = DecisionBoundaryDisplay(
        xx0=feature_1, xx1=feature_2, response=y_pred
    )
    display.plot()
    display.ax_.scatter(
        data.loc[:, "x"], data.loc[:, "y"], c=data["label"], edgecolor="black", s=10
    )
    plt.savefig(f"{name}.boundary.png", dpi=100)
    plt.close()
    
# %%
tree, df = run_dataset("D3leaves")
tree.show()

# %%
tree, df = run_dataset("D1", plot=False)
tree.show()

# %%
tree, df = run_dataset("D2", plot=False)
tree.show()
# %%
tree, df = run_dataset("D1", plot=True)
visualize_decision_boundary(tree, df, "D1")
# %%
tree, df = run_dataset("D2", plot=True)
visualize_decision_boundary(tree, df, "D2")

# %%
name = "Dbig"
df = pd.read_csv(f"data/{name}.txt", sep=" ", names=["x", "y", "label"], comment="#")
inds = np.array(range(df.shape[0]))
random.shuffle(inds)
sample_sizes = [32, 128, 512, 2048, 8192]
test_set = df.loc[inds[8192:], ]

errors = []
n_node_list = []
for dn in sample_sizes:
    sample_set = df.loc[inds[:dn], ]
    dn_name = f"Dbig.n_{dn}"
    tree, out_df = run_dataset(dn_name, df=sample_set, plot=True)
    #tree.show()
    eval_df = test_set.loc[:, ["x", "y"]]
    eval_df["pred"] = -1
    preds = tree_predict(tree, 2, eval_df)['pred'].sort_index()
    true = test_set.loc[preds.index, "label"]
    error = (true != preds).sum()/true.shape[0]
    errors += [error]
    n_nodes = len(tree) - 1
    print(f"N: {dn}, # Nodes: {n_nodes}, Error: {error}")
    n_node_list += [n_nodes]
    visualize_decision_boundary(tree, sample_set, dn_name)

plt.figure()
plt.plot(n_node_list, errors)
plt.savefig("test_error.png")

# %%
#Sklearn part
errors = []
n_node_list = []
for dn in sample_sizes:
    sample_set = df.loc[inds[:dn], ]
    dn_name = f"Dbig.n_{dn}.sklearn"
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(sample_set[["x", "y"]], sample_set["label"])
    n_nodes = clf.tree_.node_count
    eval_df = test_set.loc[:, ["x", "y"]]
    preds = clf.predict(eval_df)
    true = test_set["label"]
    error = (true != preds).sum()/true.shape[0]
    errors += [error]
    n_node_list += [n_nodes]
    print(f"N: {dn}, # Nodes: {n_nodes}, Error: {error}")
    
plt.figure()
plt.plot(n_node_list, errors)
plt.savefig("test_error.sklearn.png")

# %%
#Lagrange part
a, b, n = 0, 10, 100
points = np.random.uniform(low=a, high=b, size=n)
ys = np.sin(points)
poly = lagrange(points, ys)
train_preds = poly(points)
train_preds, ys

test_set = np.random.uniform(low=a, high=b, size=10)
test_ys = np.sin(test_set)
test_preds = Polynomial(poly.coef[::-1])(test_set)

train_err = mean_squared_error(ys, train_preds)
test_err = mean_squared_error(test_ys, test_preds)

# %%
sds = [0.01, 0.1, 1, 10]
train_errs = []
test_errs = []
for sd in sds:
    noise = np.random.normal(0, sd, n)
    points = np.random.uniform(low=a, high=b, size=n) + noise
    ys = np.sin(points)
    poly = lagrange(points, ys)
    train_preds = Polynomial(poly.coef[::-1])(points)
    test_set = np.random.uniform(low=a, high=b, size=10)
    test_ys = np.sin(test_set)
    test_preds = Polynomial(poly.coef[::-1])(test_set)
    train_err = mean_squared_error(ys, train_preds)
    test_err = mean_squared_error(test_ys, test_preds)
    train_errs += [train_err]
    test_errs += [test_err]

train_errs, test_errs

# %%
a, b, n = 0, 10, 15
points = np.random.uniform(low=a, high=b, size=n)
ys = np.sin(points)
poly = lagrange(points, ys)
train_preds = poly(points)
train_preds, ys

test_set = np.random.uniform(low=a, high=b, size=10)
test_ys = np.sin(test_set)
test_preds = Polynomial(poly.coef[::-1])(test_set)

train_err = mean_squared_error(ys, train_preds)
test_err = mean_squared_error(test_ys, test_preds)
train_err, test_err

# %%
sds = [0.01, 0.1, 1, 10]
train_errs = []
test_errs = []
for sd in sds:
    noise = np.random.normal(0, sd, n)
    points = np.random.uniform(low=a, high=b, size=n) + noise
    ys = np.sin(points)
    poly = lagrange(points, ys)
    train_preds = Polynomial(poly.coef[::-1])(points)
    test_set = np.random.uniform(low=a, high=b, size=10)
    test_ys = np.sin(test_set)
    test_preds = Polynomial(poly.coef[::-1])(test_set)
    train_err = mean_squared_error(ys, train_preds)
    test_err = mean_squared_error(test_ys, test_preds)
    train_errs += [train_err]
    test_errs += [test_err]
    
train_errs, test_errs
# %%
