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

  // const [columns, setColumn] = useState([]);
  const tmp={
    "distance": 0,
    "model": " ",
    "index": 0,
    "code": " ",
    "activity": " ",
    "level": " ",
    "credit": " "
}
  const [data, setData] = useState([tmp]);
  const defaultColumn = {
    width: "auto",
  }
  

  let queryDupak= async (e)=>{
    console.log("gonna post:",ref0.current.value);
     await fetch('http://10.242.184.93:443/search2/', {
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
         setData(responseJson.results);
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

  // let handleButtonClick= async (row)=>{
  //   console.log("gonna download:",row);
  //    await fetch('http://127.0.0.1:8000/search2/', {
  //     mode:"cors",
  //     method: 'POST',
  //     headers: {
  //       'Accept': 'application/json',
  //       "Content-type": "application/json"
  //     },
  //     // credentials: "include",
  //     body: JSON.stringify({
  //       "q": "data"
  //     })
  //   }).then((response) => response.json())// {console.log("RESPONSE0",response);})
  //      .then((responseJson) => {
  //       // responseJson=JSON.parse(responseJson)
  //        console.log("RESPONSE",responseJson.results);
  //       //  console.log("RESPONSE.results",responseJson.results);
  //        setData(responseJson.results);
  //        return responseJson;
  //      })
  //      .catch((error) => {
  //        console.error("ERROR",error);
  //      });
  // };

  let handleDownloadClick= async (idx)=>{
    let arr=idx.split("_")
    let idk=parseInt(arr[0], 10)
    let kode=arr[2]
    console.log("gonna download:",idk);

     await fetch('http://127.0.0.1:8000/download/', {
      mode:"cors",
      method: 'POST',
      headers: {
        'Accept': 'application/json',
        "Content-type": "application/json"
      },
      // credentials: "include",
      body: JSON.stringify({
        "idx": idk,
        "act":ref0.current.value,
      })
    }).then((res) => {
      console.log(res)
      return res.blob();
  })
  .then((blob) => {
      const href = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = href;
      let doc_name=kode +"_"+ref0.current.value+".docx"
      link.setAttribute('download', doc_name); //or any other extension
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
  })
  .catch((err) => {
      return Promise.reject({ Error: 'Something Went Wrong', err });
  })
  };


  const columns = useMemo(
    () => [
      {
        Header: "Kode",
        accessor: "code",
      },
      {
        Header: "Butir Kegiatan",
        accessor: "activity",
        maxWidth: 800,
        minWidth: 350,
        width: 350,
      },
      {
        Header: "Angka Kredit",
        accessor: "credit",
      },
      {
        Header: "Jenjang",
        accessor: "level",
      },
      {
        Header: "",
        accessor: "index",
        Cell: ({ cell }) => (
          <Link
          component="button"
          variant="body2"
          onClick={() => {
            handleDownloadClick(cell.value)
          }}
        >
          download
          </Link>
        )     }
     
    ],
    []
  );


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
        <Table columns={columns} data={data} />
      
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
