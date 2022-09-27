import React from "react";
import { useTable, useSortBy } from 'react-table';
import { Navigate } from 'react-router';
import { Link } from '@mui/material';
import { useMemo } from "react";




export default function Table({ data,ref0 }) {
    
  const defaultColumn = {
    width: "auto",
  }


  const handleDownloadClick= async (idx)=>{
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
          cell.value!=-1 &&
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

    // Use the useTable Hook to send the columns and data to build the table
    const {
      getTableProps, // table props from react-table
      getTableBodyProps, // table body props from react-table
      headerGroups, // headerGroups, if your table has groupings
      rows, // rows for the table based on the data passed
      prepareRow // Prepare the row (this function needs to be called for each row before getting the row props)
    } = useTable({
      columns,
      data
    },useSortBy);
  
    /* 
      Render the UI for your table
      - react-table doesn't have UI, it's headless. We just need to put the react-table props from the Hooks, and it will do its magic automatically
    */


    
    return (
      <table className="w-full text-md bg-white shadow-md rounded mb-4" {...getTableProps()} style={{ width: '100%' }}>
        <thead>
          {headerGroups.map(headerGroup => (
            <tr {...headerGroup.getHeaderGroupProps()}>
              {headerGroup.headers.map(column => (
                <th {...column.getHeaderProps(column.getSortByToggleProps())}>
                  {column.render("Header")}
                  </th>
              ))}
            </tr>
          ))}
        </thead>
        <tbody {...getTableBodyProps()}>
          {rows.map((row, i) => {
            prepareRow(row);
            return (
              <tr {...row.getRowProps()} >
                {row.cells.map(cell => {
                  return <td {...cell.getCellProps({style: { minWidth: cell.column.minWidth, width: cell.column.width,maxWidth:cell.column.maxWidth }}) } >
                    {cell.render("Cell")}
                    </td>;
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
    );
  }