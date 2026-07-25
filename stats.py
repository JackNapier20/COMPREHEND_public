# import json
# import statistics

# with open('FINAL_blossom_story_treedata.json', 'r') as f:
#    data = json.load(f)

# e1_list = [item['entail1'] for item in data if item.get('entail1') is not None]
# e2_list = [item['entail2'] for item in data if item.get('entail2') is not None]
# spec_list = [item['specificity_score'] for item in data if item.get('specificity_score') is not None]

# def calculate_metrics(values):
#     if not values:
#         return 0.0, 0.0, 0.0, 0.0
    
#     avg = sum(values) / len(values)
#     mn = min(values)
#     mx = max(values)
#     std = statistics.stdev(values) if len(values) > 1 else 0.0
    
#     return avg, mn, mx, std

# entail_1_avg, entail_1_min, entail_1_max, entail_1_stddev = calculate_metrics(e1_list)
# entail_2_avg, entail_2_min, entail_2_max, entail_2_stddev = calculate_metrics(e2_list)
# spec_avg, spec_min, spec_max, spec_stddev = calculate_metrics(spec_list)
# print(f"--- Entail 1 --- \nAvg: {entail_1_avg:.4f}, Min: {entail_1_min}, Max: {entail_1_max}, Std: {entail_1_stddev:.4f}")
# print(f"--- Entail 2 --- \nAvg: {entail_2_avg:.4f}, Min: {entail_2_min}, Max: {entail_2_max}, Std: {entail_2_stddev:.4f}")
# print(f"--- Specificity --- \nAvg: {spec_avg:.4f}, Min: {spec_min}, Max: {spec_max}, Std: {spec_stddev:.4f}")



import json
import statistics

# with open('evaltree/trees/random_evaltree_final.json', 'r') as f:
with open('Results/Gemma/FINAL_code_prompts_64.json', 'r') as f:
    data = json.load(f)
all_nodes = []
for level in data.get('levels', []):
    all_nodes.extend(level.get('nodes', []))

e1_list = [n['entail_1'] for n in all_nodes if n.get('entail_1') is not None]
e2_list = [n['entail_2'] for n in all_nodes if n.get('entail_2') is not None]
spec_list = [n['specificity'] for n in all_nodes if n.get('specificity') is not None]
f1_list = [n['faithfulness1'] for n in all_nodes if n.get('faithfulness1') is not None]
f2_list = [n['faithfulness2'] for n in all_nodes if n.get('faithfulness2') is not None]
i1_list = [n['informativeness1'] for n in all_nodes if n.get('informativeness1') is not None]
i2_list = [n['informativeness2'] for n in all_nodes if n.get('informativeness2') is not None]
sf_list = [n['specificity_faithfulness'] for n in all_nodes if n.get('specificity_faithfulness') is not None]
si_list = [n['specificity_informativeness'] for n in all_nodes if n.get('specificity_informativeness') is not None]

def calculate_metrics(values):
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    
    avg = sum(values) / len(values)
    mn = min(values)
    mx = max(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    
    return avg, mn, mx, std

e1_stats = calculate_metrics(e1_list)
e2_stats = calculate_metrics(e2_list)
spec_stats = calculate_metrics(spec_list)
f1_stats = calculate_metrics(f1_list)
f2_stats = calculate_metrics(f2_list)
i1_stats = calculate_metrics(i1_list)
i2_stats = calculate_metrics(i2_list)
sf_stats = calculate_metrics(sf_list)
si_stats = calculate_metrics(si_list)

print(f"{'Metric':<18} | {'Avg':<8} | {'Min':<8} | {'Max':<8} | {'StdDev':<8}")
print("-" * 58)
print(f"{'Entail 1':<18} | {e1_stats[0]:<8.4f} | {e1_stats[1]:<8.4f} | {e1_stats[2]:<8.4f} | {e1_stats[3]:<8.4f}")
print(f"{'Entail 2':<18} | {e2_stats[0]:<8.4f} | {e2_stats[1]:<8.4f} | {e2_stats[2]:<8.4f} | {e2_stats[3]:<8.4f}")
print(f"{'Faithfulness 1':<18} | {f1_stats[0]:<8.4f} | {f1_stats[1]:<8.4f} | {f1_stats[2]:<8.4f} | {f1_stats[3]:<8.4f}")
print(f"{'Faithfulness 2':<18} | {f2_stats[0]:<8.4f} | {f2_stats[1]:<8.4f} | {f2_stats[2]:<8.4f} | {f2_stats[3]:<8.4f}")
print(f"{'Informativeness 1':<18} | {i1_stats[0]:<8.4f} | {i1_stats[1]:<8.4f} | {i1_stats[2]:<8.4f} | {i1_stats[3]:<8.4f}")
print(f"{'Informativeness 2':<18} | {i2_stats[0]:<8.4f} | {i2_stats[1]:<8.4f} | {i2_stats[2]:<8.4f} | {i2_stats[3]:<8.4f}")
print(f"{'Specificity':<18} | {spec_stats[0]:<8.4f} | {spec_stats[1]:<8.4f} | {spec_stats[2]:<8.4f} | {spec_stats[3]:<8.4f}")
print(f"{'Spec-Faithful':<18} | {sf_stats[0]:<8.4f} | {sf_stats[1]:<8.4f} | {sf_stats[2]:<8.4f} | {sf_stats[3]:<8.4f}")
print(f"{'Spec-Informative':<18} | {si_stats[0]:<8.4f} | {si_stats[1]:<8.4f} | {si_stats[2]:<8.4f} | {si_stats[3]:<8.4f}")