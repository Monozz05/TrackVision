import { ExperimentDetailPage } from '@/components/trackvision-app'
export default async function Page({params}:{params:Promise<{experimentId:string}>}){ const {experimentId}=await params; return <ExperimentDetailPage id={experimentId}/> }
