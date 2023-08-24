import { OpenAI } from 'langchain/llms/openai'
import { HNSWLib } from 'langchain/vectorstores/hnswlib'
import { OpenAIEmbeddings } from 'langchain/embeddings/openai'

import dotenv from 'dotenv'
import fs from 'fs'
dotenv.config()

const modelName = 'text-embedding-ada-002'
const VECTOR_STORE_PATH = 'Documents.index' 
const question = "Privacy Policy"

async function run(docs) {
  const model = new OpenAI({
    modelName: modelName,
    openAIApiKey: process.env.OPENAI_API_KEY,
    temperature: 0,
  })
  let vectorStore
  if (fs.existsSync(VECTOR_STORE_PATH)) {
    // Load the existing vector store
    console.log('Loading existing vector store...')
    vectorStore = await HNSWLib.load(
      VECTOR_STORE_PATH,
      new OpenAIEmbeddings({ openAIApiKey: process.env.OPENAI_API_KEY }),
    )
    console.log('Vector store loaded.')
  } else {
    // Create vector store
    vectorStore = await HNSWLib.fromDocuments(
      docs,
      new OpenAIEmbeddings({ openAIApiKey: process.env.OPENAI_API_KEY }),
    )
    await vectorStore.save(VECTOR_STORE_PATH)
  }
  
  // response to question asked
  const response = await vectorStore.similaritySearchWithScore( // Search similar result in vector store
    question, // ask question here

  )
  console.log(response)

  if (response[0][1] < 0.2) { // Check is similarity score is less than 0.2
    return response[0][0]
  }
}

run().then((res) => {
  if (res) {
    const split = res.pageContent.split("-->") // Split question and answer from response
    console.log(split)
  } else {
    console.log('Can you please be more specific about the issue')
  }
})
