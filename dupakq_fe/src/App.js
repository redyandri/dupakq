import React, { useMemo, useRef, useEffect, useState, lazy, Suspense } from 'react'
import TextField from "@mui/material/TextField";
import "./App.css";
import Button from '@mui/material/Button';
import Box from '@mui/material/Box';
import Grid from '@mui/material/Grid';
import CircularProgress from '@mui/material/CircularProgress';
import { useCallback } from 'react';
require('dotenv').config()

const Table = lazy(() => import("./Table"));
const RGV = lazy(() => import('./RGV'));

const API_URI = 'http://' + process.env.REACT_APP_API_HOST + ':' + process.env.REACT_APP_API_PORT
// const NEO4J_URI = 'bolt://' + process.env.REACT_APP_NEO4J_HOST + ':' + process.env.REACT_APP_NEO4J_PORT;
// const NEO4J_USER = process.env.REACT_APP_NEO4J_USER;
// const NEO4J_PASSWORD = process.env.REACT_APP_NEO4J_HOST;




function App() {

  const [inputText, setInputText] = useState("");
  const ref0 = useRef();
  const [ngIsLoading, setNGIsLoading] = useState(false);

  // const [columns, setColumn] = useState([]);
  const tmp = {
    "distance": 0,
    "model": " ",
    "index": "-",
    "code": " ",
    "activity": " ",
    "level": " ",
    "credit": " "
  }
  const tmp_graph = {
    nodes: [
      { id: 1, label: "Tugas Negara" },
      { id: 2, label: "Boss!" }
    ],
    edges: [
      { id: 12, from: 1, to: 2, label: ', ' }
    ]
  }

  const [data, setData] = useState([tmp]);
  const [graph, setGraph] = useState(tmp_graph);

  const setGraphFunc = useCallback((g) => {
    setGraph(g);
  }, [graph]);

  const setGraphLoadingFunc = useCallback((g) => {
    setNGIsLoading(g);
  }, [ngIsLoading]);


  let queryDupak = async (e) => {
    // console.log("gonna post:",ref0.current.value);
    // console.log('NEO4J_URI:',NEO4J_URI)
    await fetch(API_URI + '/search2/', {
      mode: "cors",
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
        console.log("RESPONSE", responseJson.results);
        //  console.log("RESPONSE.results",responseJson.results);
        setData(responseJson.results);
        //  return responseJson;
        setGraph(tmp_graph);
      })
      .catch((error) => {
        console.error("ERROR", error);
      });
  };


  let onEnter = (e) => {
    if (e.key === 'Enter') {
      queryDupak();
    }
    // else {
    //   var lowerCase = e.target.value.toLowerCase();
    //   setInputText(lowerCase);
    // }

    // }

  };

  let onSubmit = (e) => {
    console.log("lowercase:", ref0.current.value)
    var lowerCase = ref0.current.value
    setInputText(lowerCase);
  };

  let handleDownloadClick = async (idx) => {
    let arr = idx.split("_")
    let idk = parseInt(arr[0], 10)
    let kode = arr[2]
    console.log("gonna download:", idk);

    await fetch('http://127.0.0.1:8000/download/', {
      mode: "cors",
      method: 'POST',
      headers: {
        'Accept': 'application/json',
        "Content-type": "application/json"
      },
      // credentials: "include",
      body: JSON.stringify({
        "idx": idk,
        "act": ref0.current.value,
      })
    }).then((res) => {
      console.log(res)
      return res.blob();
    })
      .then((blob) => {
        const href = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = href;
        let doc_name = kode + "_" + ref0.current.value + ".docx"
        link.setAttribute('download', doc_name); //or any other extension
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
      })
      .catch((err) => {
        return Promise.reject({ Error: 'Something Went Wrong', err });
      })
  };

  const flexStyle = {
    // border: "1px solid gray",
    margin: 1,
    flex: { xs: "100%", sm: "calc(50% - 20px)", md: "calc(33% - 20px)" }
  };

  const sx_default = {
    height: 140,
    width: 100,
    backgroundColor: (theme) =>
      theme.palette.mode === 'dark' ? '#1A2027' : '#fff',
  };

  return (
    <Box sx={{ flexGrow: 1 }}>
      <Grid container spacing={2} padding={3} paddingTop={1}>
        <Grid item xs={12} sx={{textAlign:'center'}}>
          <img src={require('./img/apple3.png')} />
        </Grid>
        <Grid item xs={11}>
          <TextField sx={flexStyle}
            id="outlined-basic"
            onKeyDown={onEnter}
            inputRef={ref0}
            fullWidth
            label="Cari Dupak"
          />
        </Grid>
        <Grid item xs={1}>
          <Button size="large"
            onClick={queryDupak}
            color="primary"
            variant='contained'
          >
            CARI AK
          </Button>
        </Grid>
        <Grid item xs={12} sx={{ border: 1, borderRadius: 1, borderColor: 'secondary.main' }}>
          <Suspense fallback={<div>Loading Table...</div>}>
            <Table sx={{ sx_default }}
              data={data}
              q={ref0}
              setGraph={setGraphFunc}
              setGraphLoading={setGraphLoadingFunc}
            />
          </Suspense>
        </Grid>
        <Grid container item xs={12} sx={{ sx_default }}>
          {ngIsLoading ? <><CircularProgress />Loading Graph ..</> : <RGV data={graph} q={ref0} />}
        </Grid>
      </Grid>
    </Box>

  );
}

export default App;
