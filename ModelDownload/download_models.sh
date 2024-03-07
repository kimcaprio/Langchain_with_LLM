# This script is used to pre=download files stored with git-lfs in CML Runtimes which do not have git-lfs support
# You can use any models that can be loaded with the huggingface transformers library. See utils/model_embedding_utls.py or utils/moderl_llm_utils.py
#EMBEDDING_MODEL_REPO="https://huggingface.co/BM-K/KoSimCSE-roberta"
#EMBDEDDING_MODEL_COMMIT="f8ef6974a7d130d24190641f5dc14c246ea85bb4"

EMBEDDING_MODEL_REPO="https://huggingface.co/jhgan/ko-sbert-nli"
EMBDEDDING_MODEL_COMMIT="b78c95e43b7ef3f8a7cca8b287367ba57516a5bf"

#LLM_MODEL_REPO="https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-GPTQ"
#LLM_MODEL_COMMIT="4e336d947ee37d99f2af735d11c4a863c74f8541"

LLM_MODEL_REPO="https://huggingface.co/ENERGY-DRINK-LOVE/eeve_dpo-v3"
LLM_MODEL_COMMIT="d9361567b4ee0a81312e0dc2e6c94ca37cb61e3c"


download_lfs_files () {
    echo "These files must be downloaded manually since there is no git-lfs here:"
    COMMIT=$1
    git ls-files | git check-attr --stdin filter | awk -F': ' '$3 ~ /lfs/ { print $1}' | while read line; do
        echo "Downloading ${line}"
        echo $(git remote get-url $(git remote))/resolve/$COMMIT/${line}
        curl -O -L $(git remote get-url $(git remote))/resolve/$COMMIT/${line}
        echo "Downloading ${line} completed"
    done
}

# Clear out any existing checked out models
rm -rf ./models
mkdir models
cd models

# Downloading model for generating vector embeddings
GIT_LFS_SKIP_SMUDGE=1 git clone ${EMBEDDING_MODEL_REPO} --branch main embedding-model 
cd embedding-model
git checkout ${EMBDEDDING_MODEL_COMMIT}
download_lfs_files $EMBDEDDING_MODEL_COMMIT
cd ..
  
# Downloading LLM model that has been fine tuned to handle instructions/q&a
GIT_LFS_SKIP_SMUDGE=1 git clone ${LLM_MODEL_REPO} --branch main llm-model
cd llm-model
git checkout ${LLM_MODEL_COMMIT}
download_lfs_files $LLM_MODEL_COMMIT
cd ..