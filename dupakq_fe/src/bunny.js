import React, { useMemo, useRef, useEffect, useState, lazy, Suspense } from 'react'
import TextField from "@mui/material/TextField";
import "./App.css";
import { Link } from '@mui/material';
import Button from '@mui/material/Button';
import Box from '@mui/material/Box';
import Stack from '@mui/material/Stack';
import Grid from '@mui/material/Grid';
import { useCallback } from 'react';
require('dotenv').config()

export default function bunny() {

  <Box sx={{ flexGrow: 1 }}>
    <Grid container spacing={2}>
      <Grid item xs={8}>
        <Graph
          graph={data}
          options={options}
          events={events}
        />
      </Grid>
      <Grid item xs={4}>
        <GraphTable
          data={graphdata}
          q={q}
        />
      </Grid>
    </Grid>
  </Box>

}

