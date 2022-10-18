import React from 'react'
import { useState } from "react";
import TextField from "@mui/material/TextField";
import "./App.css";
import {useRef, useEffect} from 'react';
import { useMemo } from "react";
import Table from "./Table";
import { Link } from '@mui/material';
import { NeoGraph, ResponsiveNeoGraph } from "./NeoGraph";

const NEO4J_URI = "bolt://10.242.184.93:7687";
const NEO4J_USER = "neo4j";
const NEO4J_PASSWORD = "test";


function App() {

  const [inputText, setInputText] = useState("");
  const ref0 = useRef();
  const empty_distance=0.9999999613152725;

  // const [columns, setColumn] = useState([]);
  const tmp={
    "distance": 0,
    "model": " ",
    "index": -1,
    "code": " ",
    "activity": " ",
    "level": " ",
    "credit": " "
}

const empty_result={
  "distance": 0,
  "model": " ",
  "index": -1,
  "code": " ",
  "activity": "Not Found",
  "level": " ",
  "credit": " "
}
  const [data, setData] = useState([tmp]);
  const defaultColumn = {
    width: "auto",
  }
  

  let queryDupak= async (e)=>{
    console.log("gonna post:",ref0.current.value);
     await fetch('http://127.0.0.1:8000/search2/', {
      mode:"cors",
      method: 'POST',
      headers: {
        'Accept': 'application/json',
        "Content-type": "application/json"
      },
      // credentials: "include",
      body: JSON.stringify({
        "q": ref0.current.value
      })
    }).then((response) => response.json())// {console.log("RESPONSE0",response);})
       .then((responseJson) => {
        // responseJson=JSON.parse(responseJson)
         console.log("RESPONSE",responseJson.results);
        //  console.log("RESPONSE.results",responseJson.results);
        let res=responseJson.results
        let d0=res[0]["distance"]
        if(d0==empty_distance){
          setData([empty_result]);
        }else{
          setData(responseJson.results);
        }
         
        //  return responseJson;
       })
       .catch((error) => {
         console.error("ERROR",error);
       });
  };


  let onEnter = (e) => {
    if (e.key === 'Enter') {
      queryDupak();
    }
    else{
      var lowerCase = e.target.value.toLowerCase();
      setInputText(lowerCase);
    }
    
    // }
    
  };

  let onSubmit=(e)=>{
    console.log("lowercase:",ref0.current.value)
    var lowerCase = ref0.current.value
    setInputText(lowerCase);
  };

  return (
    <div className="main">
      <h1>tugas negara, bos.</h1>
      <div className="search">
     
        <TextField
          id="outlined-basic"
          onKeyDown={onEnter}
          inputRef={ref0}
          variant="outlined"
          fullWidth
          label="Search"
        />
        <button type="Submit" onClick={queryDupak}>cari</button>
        <Table data={data} ref0={ref0}/>
      
      </div>
      <div className="App" style={{ fontFamily: "Quicksand" }}>
      <h1>React Neovis Example</h1>
      <ResponsiveNeoGraph
        containerId={"id0"}
        neo4jUri={NEO4J_URI}
        neo4jUser={NEO4J_USER}
        neo4jPassword={NEO4J_PASSWORD}
      />
      <NeoGraph
        width={600}
        height={500}
        containerId={"id1"}
        neo4jUri={NEO4J_URI}
        neo4jUser={NEO4J_USER}
        neo4jPassword={NEO4J_PASSWORD}
        backgroundColor={"#b2beb5"}
      />
    </div>
    </div>
  );
}

export default App;