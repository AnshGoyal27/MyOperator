import { OpenAI } from "langchain/llms/openai";
import { HNSWLib } from "langchain/vectorstores/hnswlib";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import dotenv from "dotenv";
import { Document } from "langchain/document";
import xlsx from "xlsx";
import { Tiktoken } from "@dqbd/tiktoken/lite";
import { load } from "@dqbd/tiktoken/load";
import registry from "@dqbd/tiktoken/registry.json" assert { type: "json" };
import models from "@dqbd/tiktoken/model_to_encoding.json" assert { type: "json" };
import fs from "fs";
dotenv.config();

const modelName = "text-embedding-ada-002";
const VECTOR_STORE_PATH = "Documents.index";

// function to read xlsx file and create the data format for vectorstore
async function loadFile() {
  const file = xlsx.readFile("./MyOperatorSupportFAQ.xlsx");
  let data = [];
  const sheets = file.SheetNames;
  for (let i = 0; i < 1; i++) {
    const temp = xlsx.utils.sheet_to_json(file.Sheets[file.SheetNames[i]]);
    temp.forEach((res, index) => {
      if (index <= 671) {
        data.push(
          new Document({
            pageContent: res["Article Title"] + " --> " + res["Answer\n"],
            metadata: { id: index },
          })
        );
      }
    });
  }
  return data;
}

// function to calculate cost for llm
async function cost(docs) {
  const modelKey = models[modelName];
  const model = await load(registry[modelKey]);
  const encoder = new Tiktoken(
    model.bpe_ranks,
    model.special_tokens,
    model.pat_str
  );
  const tokens = encoder.encode(JSON.stringify(docs));
  const tokenCount = tokens.length;
  const ratePerThousandTokens = 0.0004;
  const cost = (tokenCount / 1000) * ratePerThousandTokens;
  encoder.free();
  console.log(cost);
  return docs;
}

// function to create vector and load it if exists
async function run(docs) {
  const model = new OpenAI({
    modelName: modelName,
    openAIApiKey: process.env.OPENAI_API_KEY,
    temperature: 0,
  });
  console.log("Creating new vector store...");
  let vectorStore;
  // checks if vector store exists
  if (fs.existsSync(VECTOR_STORE_PATH)) {
    console.log("Loading existing vector store...");
    vectorStore = await HNSWLib.load(
      VECTOR_STORE_PATH,
      new OpenAIEmbeddings({ openAIApiKey: process.env.OPENAI_API_KEY })
    );
    console.log("Vector store loaded.");
  }
  // create vector store if not exists
  else {
    vectorStore = await HNSWLib.fromDocuments(
      docs,
      new OpenAIEmbeddings({ openAIApiKey: process.env.OPENAI_API_KEY })
    );
    await vectorStore.save(VECTOR_STORE_PATH);
    console.log("Vector Store Created");
  }
}

// run our functions
loadFile().then((docs) => cost(docs).then((docs) => run(docs)));
