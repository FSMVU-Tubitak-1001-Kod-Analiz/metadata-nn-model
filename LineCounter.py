import pandas as pd
from utils import standardize_labels


# CSV dosyasını yükle
file_path = "merged.csv"  # CSV dosyasının tam yolu
df = pd.read_csv(file_path)




def findLineCount(column_name):
    total_line_count = 0
    for content in df[column_name] :
        if isinstance(content, str):  # İçeriğin metin olup olmadığını kontrol et
            total_line_count += content.count('\n') + 1  # Satır sayısını hesapla
    return total_line_count

def filterForEqualLabelCount(df):     
    
    labels = ['bug', 'enhancement', 'performance', 'refactor', 'clean']
    df = standardize_labels(df, labels)

    # Test için veri setini dengeleme
    label_enh = df[df['labels'] == 'enhancement']
    label_bug = df[df['labels'] == 'bug']
    label_cln = df[df['labels'] == 'clean']

    min_samples = min(len(label_enh), len(label_bug), len(label_cln))
    print (min_samples)
    sampled_enh = label_enh.sample(n=min_samples, random_state=1)
    sampled_bug = label_bug.sample(n=min_samples, random_state=1)
    sampled_cln = label_cln.sample(n=min_samples, random_state=1)
    df = pd.concat([sampled_enh, sampled_bug, sampled_cln])
    return df


df = df[df['full_repo_name'].str.contains("apache", case=False, na=False)]
#df = filterForEqualLabelCount(df)

# 'repo_name' sütunundaki farklı repo isimlerini bulma
unique_repos = df['full_repo_name'].nunique()
print(f"full_repo_name count {unique_repos} ")
unique_files = df['file_path'].nunique()
print(f"different file name count {unique_files} ")

count = findLineCount("patch")
print(f"patch: {count}")

count = findLineCount("content_before")
print(f"content_before: {count } ")

count = findLineCount("content_after")
print(f"content_after: {count } ")


