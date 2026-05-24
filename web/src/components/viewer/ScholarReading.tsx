import { AgentReadingPanel } from '@/components/viewer/AgentReadingPanel'

type Props = { segmentId: string }

export function ScholarReading({ segmentId }: Props) {
  return <AgentReadingPanel segmentId={segmentId} />
}
